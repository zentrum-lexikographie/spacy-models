import bz2
import gzip
import random
import re
from collections import Counter
from io import TextIOWrapper
from pathlib import Path
from urllib.request import urlopen

import jsonstream

from .util import read_tgz, read_zip


def tag_mapping(mapping, pattern="(%s)"):
    pattern = re.compile(pattern % "|".join(map(re.escape, mapping.keys())))
    return lambda tag: pattern.sub(lambda m: mapping[m.group()], tag)


def open_text(fh, encoding="utf-8"):
    return TextIOWrapper(fh, encoding=encoding)


class TestSplitDistributor:
    def __init__(self, write):
        self.write = write
        self.n = 0

    def __call__(self, corpus, split, sentence):
        self.n += 1
        split = ("train" if (self.n % 2) == 1 else "dev") if split == "test" else split
        self.write(corpus, split, sentence)


def germeval2014(write):
    url = "https://codeload.github.com/davidsbatista/NER-datasets/zip/refs/heads/master"

    tag_suffix = re.compile(r"[BI]-(LOC|PER|OTH|ORG)(deriv|part)")
    other_tag = re.compile(r"\bOTH\b")

    def extract(zf):
        write_ = TestSplitDistributor(write)
        for split in ("train", "dev", "test"):
            name = f"NER-datasets-master/GermEval2014/NER-de-{split}.tsv"
            with open_text(zf.open(name)) as f:
                sentence = []
                for line in f:
                    if line.startswith("#"):
                        continue
                    elif not line.strip():
                        if sentence:
                            write_("germeval2014", split, tuple(sentence))
                            sentence = []
                    else:
                        line = line.split()
                        form, tag = line[1:3]
                        tag = tag_suffix.sub(r"O", tag)
                        tag = other_tag.sub("MISC", tag)
                        sentence.append((form, tag))
                if sentence:
                    write_("germeval2014", split, tuple(sentence))

    read_zip(url, extract)


def smartdata(write):
    url = "https://codeload.github.com/DFKI-NLP/smartdata-corpus/zip/refs/heads/master"

    symbols_emojis = re.compile(
        "[\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-"
        "\U0001F9FF\U0001FA70-\U0001FAFF\U0001F300-\U0001F5FF]",
        re.UNICODE,
    )

    tags = {
        "DATE": "O",
        "DISASTER_TYPE": "O",
        "DISTANCE": "O",
        "DURATION": "O",
        "LOCATION": "LOC",
        "LOCATION_CITY": "LOC",
        "LOCATION_ROUTE": "O",
        "LOCATION_STOP": "LOC",
        "LOCATION_STREET": "LOC",
        "NUMBER": "O",
        "ORGANIZATION": "ORG",
        "ORGANIZATION_COMPANY": "ORG",
        "ORG_POSITION": "O",
        "PERSON": "PER",
        "TIME": "O",
        "TRIGGER": "O",
    }

    def extract(zf):
        write_ = TestSplitDistributor(write)
        for split in ("train", "dev", "test"):
            data = zf.read(f"smartdata-corpus-master/v3_20200302/{split}.json.gz")
            for doc in jsonstream.loads(gzip.decompress(data)):
                text = doc["text"]["string"]
                result = []
                # skip sentence if it contains emojis and symbols
                if symbols_emojis.search(text):
                    continue
                for token in doc["tokens"]["array"]:
                    start = token["span"]["start"]
                    end = token["span"]["end"]
                    form = text[start:end]
                    form = re.sub(r"\s", "", form)
                    tag = token["ner"]["string"].split("-", 1)
                    if len(tag) == 2:
                        iob, ent = tag
                        ent = tags[ent]
                        tag = "-".join([iob, ent]) if ent != "O" else ent
                    else:
                        tag = tag[0]
                    result.append((form, tag))
                if result:
                    write_("smartdata", split, tuple(result))

    read_zip(url, extract)


def neiss(write):
    url = "https://codeload.github.com/NEISSproject/NERDatasets/zip/refs/heads/main"

    sturm_tags = tag_mapping(
        {
            "pers": "PER",
            "place": "LOC",
            "B-date": "O",
            "I-date": "O",
        }
    )
    arendt_tags = tag_mapping(
        {
            "I-date": "O",
            "B-date": "O",
            "person": "PER",
            "ethnicity": "MISC",
            "organization": "ORG",
            "place": "LOC",
            "event": "MISC",
            "I-language": "O",
            "B-language": "O",
        }
    )

    def extract(zf):
        write_ = TestSplitDistributor(write)
        for split in ("train", "test", "dev"):
            for text, tags in (("Sturm", sturm_tags), ("Arendt", arendt_tags)):
                name = f"NERDatasets-main/{text}/{split}_{text.lower()}.conll"
                with open_text(zf.open(name)) as f:
                    sentence = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            if sentence:
                                write_("neiss", split, tuple(sentence))
                                sentence = []
                            continue
                        form, tag = line.split(" ", 1)
                        tag = tags(tag)
                        sentence.append((form, tag))
                    if sentence:
                        write_("neiss", split, tuple(sentence))

    read_zip(url, extract)


def his_german_ner(write):
    write_ = TestSplitDistributor(write)
    for split in ("train", "dev", "test"):
        url = (
            "https://huggingface.co/datasets/stefan-it/HisGermaNER/"
            f"resolve/main/splits/HisGermaNER_v0_{split}.tsv"
        )
        with open_text(urlopen(url)) as f:
            sentence = []
            for line in f:
                if (
                    line.startswith("#")
                    or line.startswith("TOKEN")
                    or "DOCSTART" in line
                ):
                    continue
                elif not line.strip():
                    if sentence:
                        write_("his_german", split, tuple(sentence))
                        sentence = []
                else:
                    sentence.append(tuple(line.split()[0:2]))
            if sentence:
                write_("his_german", split, tuple(sentence))


def clef_hipe(write):
    url = "https://codeload.github.com/impresso/CLEF-HIPE-2020/zip/refs/heads/master"

    tags = tag_mapping(
        {
            "loc": "LOC",
            "pers": "PER",
            "org": "ORG",
            "prod": "MISC",
            "time": "O",
            "date": "O",
        }
    )

    def extract(zf):
        write_ = TestSplitDistributor(write)
        for split in ("train", "dev", "test"):
            name = f"CLEF-HIPE-2020-master/data/v1.4/de/HIPE-data-v1.4-{split}-de.tsv"
            with open_text(zf.open(name)) as f:
                sentence = []
                for line in f:
                    if (
                        line.startswith("TOKEN")
                        or line.startswith("#")
                        or not line.strip()
                    ):
                        continue
                    token = line.strip().split()
                    form, tag = token[:2]
                    tag = re.sub(r"[BI]-O\b", "O", tags(tag))
                    sentence.append((form, tag))
                    misc = set(token[9].split("|"))
                    if "PySBDSegment" in misc:
                        if sentence:
                            write_("clef_hipe", split, tuple(sentence))
                            sentence = []
                if sentence:
                    write_("clef_hipe", split, tuple(sentence))

    read_zip(url, extract)


def mobie(write):
    url = (
        "https://github.com/DFKI-NLP/MobIE/raw/refs/heads/master/"
        "v1_20210811/ner_conll03_formatted.zip"
    )

    tags = tag_mapping(
        {
            "date": "O",
            "disaster-type": "O",
            "distance": "O",
            "duration": "O",
            "location": "LOC",
            "location-city": "LOC",
            "location-route": "O",
            "location-stop": "LOC",
            "location-street": "LOC",
            "number": "O",
            "organization": "ORG",
            "organization-company": "ORG",
            "org-position": "O",
            "person": "PER",
            "time": "O",
            "trigger": "O",
            "event-cause": "O",
            "money": "O",
            "percent": "O",
            "set": "O",
        },
        r"(%s)(?!-)",
    )

    def extract(zf):
        write_ = TestSplitDistributor(write)
        for split in ("train", "dev", "test"):
            with open_text(zf.open(f"{split}.conll2003")) as f:
                sentence = []
                for line in f:
                    if line.startswith("-DOCSTART-"):
                        continue
                    if not line.strip():
                        if sentence:
                            write_("mobie", split, tuple(sentence))
                            sentence = []
                    else:
                        token = line.strip().split("\t")
                        form = token[0]
                        tag = re.sub(r"[BI]-O\b", "O", tags(token[-1]))
                        sentence.append((form, tag))
                if sentence:
                    write_("mobie", split, tuple(sentence))

    read_zip(url, extract)


def newseye(write):
    url = "https://zenodo.org/records/4573313/files/NewsEye-GT-NER_EL_StD-v1.zip"

    def extract(zf):
        write_ = TestSplitDistributor(write)
        for split in ("train", "dev", "test"):
            name = f"NewsEye-GT-NER_EL_StD-v1/NewsEye-German/{split}.tsv"
            with open_text(zf.open(name)) as f:
                sentence = []
                for line in f:
                    if line.startswith("Token\tTag\t") or line.startswith("#"):
                        continue
                    if not line.strip():
                        if sentence:
                            write_("newseye", split, tuple(sentence))
                            sentence = []
                    else:
                        token = line.strip().split("\t", 2)
                        form, tag, _ = token
                        tag = "O" if tag.endswith("HumanProd") else tag
                        sentence.append((form, tag))
                if sentence:
                    write_("newseye", split, tuple(sentence))

    read_zip(url, extract)


def conll_2003(write):
    url = "https://codeload.github.com/MaviccPRP/ger_ner_evals/zip/refs/heads/master"
    updates_url = "https://www.clips.uantwerpen.be/conll2003/ner.tgz"

    updates = []

    def extract_updates(tf):
        name = "ner/etc.2006/tags.deu"
        with open_text(tf.extractfile(name), "iso-8859-1") as f:
            for line in f:
                updates.append(line.strip())

    def extract(zf):
        updated = iter(updates)
        splits = (("train", "train"), ("testa", "dev"), ("testb", "test"))
        for suffix, split in splits:
            name = f"ger_ner_evals-master/corpora/conll2003/deu.{suffix}"
            with open_text(zf.open(name), "iso-8859-1") as f:
                sentence = []
                for line in f:
                    update = next(updated)
                    if line.startswith("-DOCSTART-"):
                        continue
                    if not line.strip():
                        assert update == ""
                        if sentence:
                            write("conll03", split, tuple(sentence))
                            sentence = []
                    else:
                        form, _ = line.split(maxsplit=1)
                        _, _, _, tag = update.split()
                        sentence.append((form, tag))
                if sentence:
                    write("conll03", split, tuple(sentence))

    read_tgz(updates_url, extract_updates)
    read_zip(url, extract)


def wikiner(write, n_ner_sentences):
    url = (
        "https://github.com/dice-group/FOX/raw/refs/tags/v2.3.0/"
        "input/Wikiner/aij-wikiner-de-wp3.bz2"
    )
    write_ = TestSplitDistributor(write)
    with bz2.open(urlopen(url), "rt", encoding="utf-8") as f:
        sentences = []
        for line in f:
            if not line.strip():
                continue
            sentences.append(
                tuple((t[0], t[2]) for t in (t.split("|", 2) for t in line.split()))
            )

        random.seed(0)
        random.shuffle(sentences)
        sentences = sentences[:n_ner_sentences]

        dev_size = test_size = int(0.1 * len(sentences))
        train_size = (len(sentences) - dev_size) - test_size

        for sentence in sentences[:train_size]:
            write_("wikiner", "train", sentence)
        for sentence in sentences[train_size : train_size + dev_size]:
            write_("wikiner", "dev", sentence)
        for sentence in sentences[train_size + dev_size :]:
            write_("wikiner", "test", sentence)


project_dir = Path(__file__).parent.parent
dataset_dir = project_dir / "dataset"


def open_ner_split(split):
    return bz2.open(dataset_dir / f"ner-d.{split}.tsv.bz2", "wt", encoding="utf-8")


def main():
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ner_splits = {k: open_ner_split(k) for k in ("train", "dev", "test")}
    ner_splits_n = Counter()

    def write_ner(corpus, split, sentence):
        invalid = False
        for form, _ in sentence:
            if '"' in form or "'" in form:
                invalid = True
                break
        if invalid:
            return

        ner_splits_n[split] += 1
        ner_splits[split].write("\n".join(("\t".join(t) for t in sentence)) + "\n\n")

    germeval2014(write_ner)
    smartdata(write_ner)
    neiss(write_ner)
    his_german_ner(write_ner)
    clef_hipe(write_ner)
    mobie(write_ner)
    newseye(write_ner)
    conll_2003(write_ner)
    wikiner(write_ner, ner_splits_n.total())

    for f in ner_splits.values():
        f.close()


if __name__ == "__main__":
    main()
