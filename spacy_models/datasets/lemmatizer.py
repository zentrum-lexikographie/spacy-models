import argparse
import re
from itertools import islice
from pathlib import Path
from random import shuffle

import dwdsmor
import dwdsmor.tag.hdt
import lxml.etree as ET
import spacy
import spacy.tokens
import spacy.vocab
from tqdm import tqdm


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


def wb_examples(wb_xml_file):
    doc = ET.parse(wb_xml_file)
    return (tree_text(e) for e in doc.iter(qn("Belegtext")))


def extract_wb_examples(wb_dir):
    wb_xml_files = [str(f) for f in wb_dir.rglob("*.xml")]
    shuffle(wb_xml_files)
    for wb_xml_file in wb_xml_files:
        for sentence in wb_examples(wb_xml_file):
            yield sentence


def annotate_examples(examples):
    return spacy.load("de_zdl_lg").pipe(examples)


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


def trim_spacy_doc(vocab, doc):
    return spacy.tokens.Doc(
        vocab=vocab,
        words=[t.text for t in doc],
        spaces=[(True if t.whitespace_ else False) for t in doc],
        lemmas=[t.lemma_ for t in doc],
    )


def trim_spacy_docs(docs):
    vocab = spacy.vocab.Vocab()
    return tuple(trim_spacy_doc(vocab, d) for d in docs)


arg_parser = argparse.ArgumentParser(description="Create Lemmatizer training dataset")
arg_parser.add_argument(
    "-d",
    "--dwdswb-dir",
    help="directory with DWDS-WB sources",
    type=Path,
    required=True,
)
arg_parser.add_argument(
    "-n",
    "--num-examples",
    help="number of example sentences to extract",
    type=int,
    default="1000000",
)
arg_parser.add_argument(
    "-o",
    "--output-dir",
    help="output directory with training dataset in spaCy format",
    type=Path,
    required=True,
)


def main():
    args = arg_parser.parse_args()
    examples = extract_wb_examples(args.dwdswb_dir)
    examples = annotate_examples(examples)
    examples = lemmatize_examples(examples)

    output_dir = args.output_dir
    assert not output_dir.is_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train, dev, test = [], [], []

    total = args.num_examples
    progress = tqdm(
        desc="Extracting",
        unit=" sentences",
        unit_scale=True,
        smoothing=0.01,
        total=total,
    )
    for ei, example in enumerate(islice(examples, total), 0):
        bm = ei % 10
        bucket = train if bm < 8 else dev if bm < 9 else test
        bucket.append(example)
        progress.update(1)
    for split, docs in zip(("train", "dev", "test"), (train, dev, test)):
        docs = trim_spacy_docs(docs)
        doc_bin = spacy.tokens.DocBin(docs=docs, store_user_data=True)
        (output_dir / f"dwdswb.{split}.spacy").write_bytes(doc_bin.to_bytes())


if __name__ == "__main__":
    main()
