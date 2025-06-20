# spacy-models

_German spaCy models trained on German UD-HDT and a collection of German NER datasets_

This project trains a part-of-speech tagger, morphologizer,
lemmatizer, dependency parser and NER tagger from the German UD-HDT
and a collection of German NER datasets. It takes care of

1. data preparation,
1. converting it to spaCy's format,
1. training separate models optimized for CPU as well as GPU architectures, and
1. evaluating the trained models.

Note that multi-word tokens will be merged together when the corpus is
converted since spaCy does not support multi-word token expansion.

## Installation

Initialize/update environment:

``` shell
pip install -U pip pip-tools setuptools
```

For training on a GPU system (recommended):

```shell
pip install -e .
```

For development:

```shell
pip install -e .[dev]
```

## Training

``` shell
GPU_ID=0 spacy-models-build
```

Including release to HuggingFace:

``` shell
HF_PUSH=1 GPU_ID=0 spacy-models-build
```

## Updating NER dataset

``` shell
spacy-models-dataset
```

## Datasets

* E. B. Völker, M. Wendt, F. Hennig, and A. Köhn (2019). HDT-UD: A very large Universal Dependencies Treebank for German. Proceedings of the Third Workshop on Universal Dependencies (UDW, SyntaxFest 2019), pages 46–57, Paris, France. Association for Computational Linguistics.
* D. Benikova, C. Biemann, M. Reznicek (2014). NoSta-D Named Entity Annotation for German: Guidelines and Dataset. Proceedings of LREC 2014, Reykjavik, Iceland.
* M. Schiersch, V. Mironova, M. Schmitt, P. Thomas, A. Gabryszak, L. Hennig (2018). A German Corpus for Fine-Grained Named Entity Recognition and Relation Extraction of Traffic and Industry Events. Proceedings of LREC 2018, Miyazaki, Japan.
* J. Zöllner, K. Sperfeld, C. Wick, R. Labahn (2021). Optimizing Small BERTs Trained for German NER. Information 2021, 12, 443.
* M. Ehrmann, M. Romanello, A. Flückiger, and S. Clematide (2020). Extended Overview of CLEF HIPE 2020: Named Entity Processing on Historical Newspapers in Working Notes of CLEF 2020 - Conference and Labs of the Evaluation Forum, Thessaloniki, Greece, 2020, vol. 2696, p. 38. doi: 10.5281/zenodo.4117566.
* L. Hennig, P. T. Truong, A. Gabryszak (2021). Mobie: A German Dataset for Named Entity Recognition, Entity Linking and Relation Extraction in the Mobility Domain. arXiv preprint arXiv:2108.06955.
* A. Hamdi, E. Linhares Pontes, E. Boros, T. T. H. Nguyen, G. Hackl, J. G. Moreno, A. Doucet (2021). Multilingual Dataset for Named Entity Recognition, Entity Linking and Stance Detection in Historical Newspapers (V1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4573313
* J. Nothman, N. Ringland, W. Radford, T. Murphy, J. R. Curran (2013). Learning Multilingual Named Entity Recognition from Wikipedia. Artificial Intelligence, 194, 151-175.
* S. Schweter (2025). HisGermaNER (Revision 83571b3). doi: 10.57967/hf/5770, https://huggingface.co/datasets/stefan-it/HisGermaNER
