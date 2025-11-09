import bz2
import json
import os
from collections import namedtuple
from pathlib import Path
from shutil import rmtree
from subprocess import check_call

import spacy
from spacy.cli.download import download
from spacy.cli.evaluate import evaluate
from spacy.cli.init_config import init_config
from spacy.cli.package import package
from spacy.cli.train import train
from spacy.tokens import DocBin
from spacy.training.converters import conll_ner_to_docs, conllu_to_docs

from .util import read_zip
from .version import __version__

is_release = os.environ.get("ZDL_RELEASE", "")
gpu_id = int(os.environ.get("GPU_ID", "0"))

project_dir = Path(__file__).parent.parent
dataset_dir = project_dir / "dataset"
configs_dir = project_dir / "configs"
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


def install_base_model():
    try:
        spacy.load("de_core_news_lg")
    except Exception:
        download("de_core_news_lg")


def clean_output_dirs():
    for d in (configs_dir, training_dir, packages_dir):
        if d.is_dir():
            rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


pipeline = ["tagger", "morphologizer", "parser", "ner", "trainable_lemmatizer"]


def configure_base(gpu, component_names, base_model):
    config = init_config(lang="de", pipeline=pipeline, optimize="accuracy", gpu=gpu)

    # fill config
    nlp = spacy.util.load_model_from_config(config, auto_fill=True, validate=False)
    nlp = spacy.util.load_model_from_config(nlp.config)
    config = nlp.config

    if gpu:
        transformer_model = config["components"]["transformer"]["model"]
        transformer_model["name"] = "distilbert-base-german-cased"

    score_weights = config["training"]["score_weights"]
    for weight in ("ents_f", "ents_p", "ents_r", "lemma_acc"):
        score_weights[weight] = 0.0

    return config


def configure_lemmatizer(gpu, component_names, base_model):
    nlp = spacy.load(base_model)

    if not gpu:
        # retrain lemmatizer with CPU-optimized defaults
        nlp.remove_pipe("trainable_lemmatizer")
        nlp.add_pipe("trainable_lemmatizer")

    config = nlp.config.copy()

    if gpu:
        config["training"]["annotating_components"] = ["transformer"]

    components = config["components"]
    frozen = config["training"]["frozen_components"] = []
    for c in component_names:
        if c != "trainable_lemmatizer":
            components[c] = {"source": base_model}
            frozen.append(c)

    # TODO
    # https://github.com/jmyerston/greCy/blob/main/configs/lemmatizer_trf.cfg
    #
    # lemmatizer = components["trainable_lemmatizer"]
    # lemmatizer["min_tree_freq"] = 1
    # lemmatizer["top_k"] = 7
    # lemmatizer["overwrite"] = True

    score_weights = config["training"]["score_weights"] = {}
    score_weights["lemma_acc"] = 1.0

    return config


def configure_ner(gpu, component_names, base_model):
    config = spacy.load(base_model).config.copy()

    components = config["components"]
    frozen = config["training"]["frozen_components"] = []
    for c in component_names:
        component = components[c] = {"source": base_model}
        if c == "ner":
            component["replace_listeners"] = ["model.tok2vec"]
        else:
            frozen.append(c)

    score_weights = config["training"]["score_weights"] = {}
    score_weights["ents_f"] = 1.0

    return config


def perf_base(perf, base_perf):
    return base_perf


def perf_lemma(perf, lemma_perf):
    perf["lemma_acc"] = lemma_perf["lemma_acc"]
    return perf


def perf_ner(perf, ner_perf):
    perf["ents_p"] = ner_perf["ents_p"]
    perf["ents_r"] = ner_perf["ents_r"]
    perf["ents_f"] = ner_perf["ents_f"]
    return perf


TrainingStage = namedtuple("TrainingStage", ["corpus", "configure", "merge_perf"])

training_stages = [
    TrainingStage("hdt", configure_base, perf_base),
    TrainingStage("dwdswb", configure_lemmatizer, perf_lemma),
    TrainingStage("ner-d", configure_ner, perf_ner),
]


def train_models():
    for gpu in (True, False):
        component_names = pipeline.copy()
        component_names.insert(0, "transformer" if gpu else "tok2vec")

        model = None
        for stage in training_stages:
            id = ".".join(("gpu" if gpu else "cpu", stage.corpus))
            cfg_path = configs_dir / f"{id}.cfg"
            training_path = training_dir / id
            result_path = training_path / "model-best"
            if result_path.is_dir():
                model = str(result_path)
                continue

            config = stage.configure(gpu, component_names, model)
            config["paths"]["train"] = f"dataset/{stage.corpus}.train.spacy"
            config["paths"]["dev"] = f"dataset/{stage.corpus}.dev.spacy"
            if model:
                config["initialize"]["before_init"] = {
                    "@callbacks": "spacy.copy_from_base_model.v1",
                    "tokenizer": model,
                    "vocab": model,
                }

            config.to_disk(cfg_path)
            train(cfg_path, training_path, use_gpu=gpu_id)
            model = str(result_path)

        perf = None
        for stage in training_stages:
            perf = stage.merge_perf(
                perf,
                evaluate(
                    model,
                    dataset_dir / f"{stage.corpus}.test.spacy",
                    silent=False,
                    use_gpu=gpu_id,
                ),
            )
        meta_path = Path(model) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["performance"] = perf
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        package(
            Path(model),
            packages_dir,
            name=("zdl_" + ("dist" if gpu else "lg")),
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
    install_base_model()
    clean_output_dirs()
    train_models()
    release()


if __name__ == "__main__":
    main()
