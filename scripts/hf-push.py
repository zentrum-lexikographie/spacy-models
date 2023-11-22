#!/usr/bin/env python

import os
from pathlib import Path

import spacy_huggingface_hub

ns = 'zentrum-lexikographie'
version = os.environ['VERSION']
packages = (Path(__file__) / '..' / '..' / 'packages').resolve()

for anno_type in ['dep_hdt', 'ner']:
    for model_type in ['lg', 'dist']:
        k = f'de_dwds_{anno_type}_{model_type}-{version}'
        whl = packages / k / 'dist' / f'{k}-py3-none-any.whl'
        spacy_huggingface_hub.push(whl.as_posix(), namespace=ns)
