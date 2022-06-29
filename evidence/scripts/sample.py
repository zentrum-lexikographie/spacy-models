import itertools
import random

import click
import conllu
from conllu import TokenList
from tqdm import tqdm


def convert_ner(s):
    ne_type, ne_class = s.split('-')
    if ne_type in "BS":
        return f"B-{ne_class}"
    else:
        return f"I-{ne_class}"


def is_valid_sentence(s: TokenList) -> bool:
    def sent_filter_length(sentence) -> bool:
        return 5 <= len(sentence) <= 100

    def sent_filter_tags(sentence) -> bool:
        return any(t['upos'] in ["NOUN", "VERB", "AUX"] for t in sentence)

    def sent_filter_invalid_tags(sentence) -> bool:
        return sum(t['upos'] in {'NUM', 'PUNCT'} for t in sentence) < (len(sentence) // 2)

    return all([
        sent_filter_length(s),
        sent_filter_tags(s),
        sent_filter_invalid_tags(s)
    ])


def get_sentence_iter(files, sample_ratio=0.5):
    conll_files = [conllu.parse_incr(open(fin), fields=conllu.parser.DEFAULT_FIELDS) for fin in files]
    finished_files = []
    document_meta = {}
    for conll_i, conll_file in itertools.cycle(enumerate(conll_files)):
        if conll_i == 0:
            finished_files = [False for _ in range(len(conll_files))]
        try:
            sent = next(conll_file)
            if "DDC:meta.file_" in sent.metadata:
                document_meta[conll_i] = {
                    'corpus_id': sent.metadata.get("DDC:meta.collection"),
                    'basename': sent.metadata.get("DDC:meta.basename"),
                    'date': sent.metadata.get("DDC:meta.date_"),
                }

            if not is_valid_sentence(sent):
                continue
            if random.random() < sample_ratio:
                sent.metadata.clear()
                sent.metadata['corpus_id'] = document_meta[conll_i]['corpus_id']
                sent.metadata['basename'] = document_meta[conll_i]['basename']
                sent.metadata['date'] = document_meta[conll_i]['date']
                for t in sent:
                    if 'misc' in t and 'NER' in t['misc']:
                        t['misc']['NE'] = convert_ner(t['misc']['NER'])
                        del t['misc']['NER']
                yield sent
        except StopIteration:
            finished_files[conll_i] = True
        if all(finished_files):
            break


@click.command()
@click.argument('inputs', nargs=-1, required=True, type=str)
@click.option('-o', '--output', default='-', type=click.File('w'))
@click.option('-s', '--sample-ratio', default=0.5, type=float)
def main(inputs, output, sample_ratio):
    sentences = get_sentence_iter(inputs, sample_ratio)
    for sentence in tqdm(sentences):
        output.write(sentence.serialize())
    output.flush()


if __name__ == '__main__':
    main()
