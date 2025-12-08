from spacy.tokens import DocBin
from spacy.training.converters import conllu_to_docs

from ..env import dataset_dir
from ..util import read_zip


def prepare():
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
