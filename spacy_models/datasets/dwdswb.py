import gzip
import os
import re
from functools import cache
from itertools import islice
from random import shuffle

import dwdsmor
import dwdsmor.tag.hdt
import lxml.etree as ET
import requests
import spacy
import spacy.tokens
import spacy.vocab

from ..env import dataset_dir

dataset_file = dataset_dir / "dwdswb.xml.gz"


def download_dataset():
    if dataset_file.is_file():
        return
    dataset_url = os.environ.get("DWDSWB_DATASET_URL")
    ci_job_token = os.environ.get("CI_JOB_TOKEN")
    assert dataset_url, "No $DWDSWB_DATASET_URL given"
    assert ci_job_token, "No $CI_JOB_TOKEN given"
    r = requests.get(dataset_url, headers={"JOB-TOKEN": ci_job_token}, stream=True)
    r.raise_for_status()
    with dataset_file.open("wb") as f:
        for c in r.iter_content(chunk_size=8192):
            f.write(c)


@cache
def qn(tag):
    return "{http://www.dwds.de/ns/1.0}" + tag


whitespace_run = re.compile(r"\s+")


def tree_text(tree):
    if tree.tag == qn("Streichung"):
        return ""
    text = tree.text or ""
    for child in tree:
        text += tree_text(child)
        text += child.tail or ""
    text = whitespace_run.sub(" ", text).strip()
    return text


def extract_wb_examples():
    with gzip.open(dataset_file, "rb") as f:
        doc = ET.parse(f)
        examples = list(doc.iter(qn("Belegtext")))
        shuffle(examples)
        return (tree_text(e) for e in examples)


def valid_analysis(a):
    return not any((a.orthinfo, a.syninfo, a.metainfo))


def morph(token_morph, k):
    v = ",".join(token_morph.get(k))
    return v if v else None


def analyze(analyzer, token):
    traversals = tuple(analyzer.analyze(token.text))
    traversals = tuple(filter(valid_analysis, traversals))
    if len(traversals) == 1:
        return traversals
    token_morph = token.morph
    criteria = {
        k: frozenset(v) if v else None
        for k, v in dwdsmor.tag.hdt.criteria(
            token.tag_,
            morph(token_morph, "Number"),
            morph(token_morph, "Gender"),
            morph(token_morph, "Case"),
            morph(token_morph, "Person"),
            morph(token_morph, "Tense"),
            morph(token_morph, "Degree"),
            morph(token_morph, "Mood"),
            morph(token_morph, "VerbForm"),
            None,  # TODO: separable verbs via syninfo
        ).items()
    }
    criteria_stack = list((k, v) for k, v in criteria.items() if v)
    criteria_stack.reverse()
    while criteria_stack:
        if len(traversals) == 1:
            break
        attr, attr_vals = criteria_stack.pop()
        filtered = tuple((t for t in traversals if getattr(t, attr) in attr_vals))
        traversals = filtered or traversals
    return sorted(traversals, key=lambda t: len(t.spec))


def lemmatized(analyzer, sentence):
    for token in sentence:
        analyses = analyze(analyzer, token)
        lemmata = {a.analysis for a in analyses}
        if len(lemmata) == 1:
            token.lemma_ = next(iter(lemmata))
            continue
        if token.i - token.sent.start == 0:
            lemmata = {lemma.lower() for lemma in lemmata}
            if len(lemmata) == 1:
                token.lemma_ = next(iter(lemmata))
        # only return sentences which have been fully and unambiguously lemmatized
        return None
    return sentence


def lemmatize_examples(examples):
    morphology = dwdsmor.automata("zentrum-lexikographie/dwdsmor-dwds")
    analyzer = morphology.analyzer("lemma")
    for example in examples:
        example = lemmatized(analyzer, example)
        if example:
            yield example


def condense_spacy_doc(vocab, doc):
    return spacy.tokens.Doc(
        vocab=vocab,
        words=[t.text for t in doc],
        spaces=[(True if t.whitespace_ else False) for t in doc],
        lemmas=[t.lemma_ for t in doc],
    )


def prepare(nlp):
    nlp.add_pipe("doc_cleaner")

    download_dataset()
    examples = extract_wb_examples()
    examples = nlp.pipe(examples)
    examples = lemmatize_examples(examples)

    num_examples = 1000000
    train, dev, test = [], [], []

    for ei, example in enumerate(islice(examples, num_examples), 0):
        bm = ei % 10
        bucket = train if bm < 8 else dev if bm < 9 else test
        bucket.append(example)
    for split, docs in zip(("train", "dev", "test"), (train, dev, test)):
        vocab = spacy.vocab.Vocab()
        docs = tuple(condense_spacy_doc(vocab, d) for d in docs)
        doc_bin = spacy.tokens.DocBin(docs=docs, store_user_data=True)
        (dataset_dir / f"dwdswb.{split}.spacy").write_bytes(doc_bin.to_bytes())

    nlp.remove_pipe("doc_cleaner")


if __name__ == "__main__":
    download_dataset()
