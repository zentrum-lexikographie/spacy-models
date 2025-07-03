import logging
import re
import subprocess

import spacy
import thinc.api

from .version import __version__

#  Align model versions with latest minor version of the package, so
#  updates are not required for patch releases
model_version = re.sub(r"\.[0-9]+$", ".0", __version__)


def model_package_spec(model):
    package = model.replace("_", "-")
    return (
        f"{model} @ https://repo.zdl.org/repository/pypi/packages/"
        f"{package}/{model_version}/{model}-{model_version}-py3-none-any.whl"
    )


packages = {
    model: model_package_spec(model)
    for corpus in ("hdt", "ner_d")
    for type in ("dist", "lg")
    for model in (f"de_{corpus}_{type}",)
}


logger = logging.getLogger(__name__)


def load_or_install(model):
    try:
        return spacy.load(model)
    except OSError:
        assert model in packages, model
        subprocess.check_call(("pip", "install", "-qqq", packages[model]))
        return spacy.load(model)


def load(model_type="dist", ner=True, gpu_id=None):
    assert model_type in ("dist", "lg")
    if gpu_id is not None:
        thinc.api.set_gpu_allocator("pytorch")
        thinc.api.require_gpu(gpu_id)
    nlp = load_or_install(f"de_hdt_{model_type}")
    if ner:
        ner = load_or_install(f"de_ner_d_{model_type}")
        ner.replace_listeners(
            "transformer" if model_type == "dist" else "tok2vec",
            "ner",
            ("model.tok2vec",),
        )
        nlp.add_pipe("ner", source=ner, name="ner_d")
    nlp.add_pipe("doc_cleaner")
    return nlp


__all__ = ["load"]
