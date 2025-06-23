import bz2
import json
import os
from pathlib import Path
from shutil import rmtree
from subprocess import check_call

import spacy
from spacy.cli.download import download
from spacy.cli.evaluate import evaluate
from spacy.cli.init_config import fill_config
from spacy.cli.package import package
from spacy.cli.train import train
from spacy.tokens import DocBin
from spacy.training.converters import conll_ner_to_docs, conllu_to_docs
from wasabi import msg

from .util import read_zip
from .version import __version__

is_release = os.environ.get("ZDL_RELEASE", "")
gpu_id = int(os.environ.get("GPU_ID", "-1"))

project_dir = Path(__file__).parent.parent
dataset_dir = project_dir / "dataset"
configs_dir = project_dir / "configs"
configs_filled_dir = configs_dir / "filled"
training_dir = project_dir / "training"
packages_dir = project_dir / "packages"


def prepare_hdt():
    url = (
        "https://codeload.github.com/UniversalDependencies/UD_German-HDT/"
        "zip/refs/tags/r2.16"
    )

    def read_docs(zf, name):
        with zf.open(name) as f:
            return conllu_to_docs(
                f.read().decode("utf-8"),
                n_sents=32,
                merge_subtokens=True,
                no_print=True,
            )

    def extract(zf):
        path_prefix = "UD_German-HDT-r2.16/de_hdt-ud-"
        docs = [
            d
            for p1 in ("a", "b")
            for p2 in ("1", "2")
            for d in read_docs(zf, f"{path_prefix}train-{p1}-{p2}.conllu")
        ]
        data = DocBin(docs=docs, store_user_data=True).to_bytes()
        (dataset_dir / "hdt.train.spacy").write_bytes(data)

        for split in ("dev", "test"):
            docs = read_docs(zf, f"{path_prefix}{split}.conllu")
            data = DocBin(docs=docs, store_user_data=True).to_bytes()
            (dataset_dir / f"hdt.{split}.spacy").write_bytes(data)

    read_zip(url, extract)


def prepare_ner_d():
    for split in ("train", "dev", "test"):
        file_name = f"ner-d.{split}.tsv.bz2"
        with bz2.open(dataset_dir / file_name, "rt", encoding="utf-8") as f:
            docs = conll_ner_to_docs(f.read(), n_sents=32, no_print=True)
            data = DocBin(docs=docs, store_user_data=True).to_bytes()
            (dataset_dir / f"ner-d.{split}.spacy").write_bytes(data)


def train_models():
    for d in [configs_filled_dir, training_dir, packages_dir]:
        if d.is_dir():
            rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    try:
        spacy.load("de_core_news_lg")
    except Exception:
        download("de_core_news_lg")

    for corpus in ("hdt", "ner-d"):
        for model in ("lg", "dist"):
            prefix = f"{corpus}-{model}"
            msg.divider(f"Pipeline: {prefix}")

            cfg_file_name = f"{prefix}.cfg"
            fill_config(
                configs_filled_dir / cfg_file_name,
                configs_dir / cfg_file_name,
                silent=True,
            )

            training_path = training_dir / prefix
            train(configs_filled_dir / cfg_file_name, training_path, use_gpu=gpu_id)

            model_path = training_path / "model-best"

            meta_path = model_path / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["performance"] = evaluate(
                str(model_path),
                dataset_dir / f"{corpus}.test.spacy",
                silent=False,
                use_gpu=gpu_id,
            )
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            package(
                model_path,
                packages_dir,
                name=prefix.replace("-", "_"),
                version=__version__,
                create_wheel=True,
                create_sdist=False,
                force=True,
                silent=False,
            )


def release():
    if is_release:
        for whl in packages_dir.rglob("*.whl"):
            check_call(
                (
                    "twine",
                    "upload",
                    "--repository-url",
                    "https://repo.zdl.org/repository/pypi/",
                    whl.as_posix(),
                )
            )


def main():
    prepare_hdt()
    prepare_ner_d()
    train_models()
    release()


if __name__ == "__main__":
    main()
