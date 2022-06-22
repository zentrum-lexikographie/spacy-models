import os
import random

import click
import conllu
from tqdm import tqdm


@click.command()
@click.argument('output', type=click.Path())
@click.option('-i', '--inputs', default='-', type=click.File('r'))
@click.option('--train-ratio', default=0.8, type=float)
@click.option('--dev-ratio', default=0.1, type=float)
def main(inputs, output, train_ratio, dev_ratio):
    sentences = conllu.parse_incr(inputs, fields=conllu.parser.DEFAULT_FIELDS)
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'train.conllu'), 'w') as train_out,\
        open(os.path.join(output, 'dev.conllu'), 'w') as dev_out, \
        open(os.path.join(output, 'test.conllu'), 'w') as test_out:
        for sent_i, sentence in tqdm(enumerate(sentences)):
            p = random.random()
            if p < train_ratio:
                out = train_out
            elif p < train_ratio + dev_ratio:
                out = dev_out
            else:
                out = test_out
            out.write(sentence.serialize())
            if sent_i % 1000 == 0:
                train_out.flush()
                dev_out.flush()
                test_out.flush()


if __name__ == '__main__':
    main()
