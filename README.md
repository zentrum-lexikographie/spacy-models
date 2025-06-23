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


* Benikova, Darina, Chris Biemann, und Marc Reznicek. „NoSta-D Named Entity Annotation for German: Guidelines and Dataset“. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), herausgegeben von Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, und Stelios Piperidis, 2524–31. Reykjavik, Iceland: European Language Resources Association (ELRA), 2014. https://aclanthology.org/L14-1251/.
* Borges Völker, Emanuel, Maximilian Wendt, Felix Hennig, und Arne Köhn. „HDT-UD: A very large Universal Dependencies Treebank for German“. In Proceedings of the Third Workshop on Universal Dependencies (UDW, SyntaxFest 2019), herausgegeben von Alexandre Rademaker und Francis Tyers, 46–57. Paris, France: Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/W19-8006.
* Ehrmann, Maud, Matteo Romanello, SImon Clematide, und Alex Flückiger. „CLEF-HIPE-2020 Shared Task Named Entity Datasets“. Zenodo, 11. März 2020. https://zenodo.org/records/6046853.
Hamdi, Ahmed, Elvys Linhares Pontes, Emanuela Boros, Thi Tuyet Hai Nguyen, Günter Hackl, Jose G. Moreno, und Antoine Doucet. „A Multilingual Dataset for Named Entity Recognition, Entity Linking and Stance Detection in Historical Newspapers“. Gehalten auf der The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021), 15. April 2021. https://doi.org/10.5281/zenodo.4694466.
* Hennig, Leonhard, Phuc Tran Truong, und Aleksandra Gabryszak. „MobIE: A German Dataset for Named Entity Recognition, Entity Linking and Relation Extraction in the Mobility Domain“. In Proceedings of the 17th Conference on Natural Language Processing (KONVENS 2021), herausgegeben von Kilian Evang, Laura Kallmeyer, Rainer Osswald, Jakub Waszczuk, und Torsten Zesch, 223–27. Düsseldorf, Germany: KONVENS 2021 Organizers, 2021. https://aclanthology.org/2021.konvens-1.22/.
* Nothman, Joel, Nicky Ringland, Will Radford, Tara Murphy, und James R. Curran. „Learning multilingual named entity recognition from Wikipedia“. Artificial Intelligence, Artificial Intelligence, Wikipedia and Semi-Structured Resources, 194 (1. Januar 2013): 151–75. https://doi.org/10.1016/j.artint.2012.03.006.
* Schiersch, Martin, Veselina Mironova, Maximilian Schmitt, Philippe Thomas, Aleksandra Gabryszak, und Leonhard Hennig. „A German Corpus for Fine-Grained Named Entity Recognition and Relation Extraction of Traffic and Industry Events“. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), herausgegeben von Nicoletta Calzolari, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Koiti Hasida, Hitoshi Isahara, u. a. Miyazaki, Japan: European Language Resources Association (ELRA), 2018. https://aclanthology.org/L18-1703/.
* Schweter, Stefan. „HisGermaNER (Revision 83571b3)“. Hugging Face, 2025. https://doi.org/10.57967/hf/5770.
* Tjong Kim Sang, Erik F., und Fien De Meulder. „Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition“. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, 142–47, 2003. https://aclanthology.org/W03-0419/.
* Zöllner, Jochen, Konrad Sperfeld, Christoph Wick, und Roger Labahn. „Optimizing small BERTs trained for German NER“. Information 12, Nr. 11 (25. Oktober 2021): 443. https://doi.org/10.3390/info12110443.


## License

While the code of this project is licensed under the GNU General
Public License v3, the datasets used for training and evaluating the
resulting models are licensed under different terms, most commonly
under a Creative Commons license, but with different restrictions and
conditions regarding their reuse. Therefore, before reusing the
referenced datasets, please ensure that you have the required rights
and your usage adheres to the respective license terms.
