# spacy-models-de-hdt-wikiner

_German spaCy models trained on German UD-HDT and the WikiNER corpus_

This project trains a part-of-speech tagger, morphologizer,
lemmatizer, dependency parser and NER tagger from the German UD-HDT
and WikiNER corpus.  It takes care of data preparation, converting it
to spaCy's format, training separate models for GPU (`-dist`) and CPU
(`-lg`) architectures, as well as evaluating the trained models. Note
that multi-word tokens will be merged together when the corpus is
converted since spaCy does not support multi-word token expansion.

## Installation

For training on a GPU system (recommended):

```shell
pip install -e .[gpu]
```

For CPU-based training:

```shell
pip install -e .[cpu]
```

## Training

``` shell
spacy project assets
GPU_ID=0 spacy project run all # GPU_ID=-1 for CPU-based training
```

## Pushing artifacts to HuggingFace

``` shell
./hf-push
```
