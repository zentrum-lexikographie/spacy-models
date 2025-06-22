import zdl_spacy


def test_model_loading():
    for model_type in ("dist", "lg"):
        nlp = zdl_spacy.load(model_type)
        annotated = nlp("Das ist ein Test aus Berlin!")
        assert len(annotated.ents) > 0
