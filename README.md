# spacy-models

_German spaCy models trained on UD-HDT and custom datasets for NER and lemmatization_

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15721441.svg)](https://doi.org/10.5281/zenodo.15721441)

This project trains a [spaCy](https://spacy.io/) pipeline with two
models, geared towards the **annotation of German texts for
lexicographic use cases** at the [Zentrum für digitale Lexikographie
der deutschen Sprache](https://www.zdl.org/). The pipeline comprises a
part-of-speech tagger, morphologizer, lemmatizer, syntactic dependency
parser and NER tagger. The notable differences in comparison to
spaCy's default pipeline for German are:

1. The tagger and dependency parser are trained on data from the
   [Hamburg Dependency Treebank](https://aclanthology.org/L14-1666/)
   (HDT). Its tagset, rather than the one from the TIGER corpus, is
   more amenable to ZDL-specific downstream tasks like collocation
   extraction.
1. The NER tagger is trained on an aggregated custom dataset of
   several gold and silver standards (see below for
   details). Moreover, it is an integral part of the pipeline for CPU
   and GPU architectures because recognizing named entities in
   examined textual evidence is a central capability in the context of
   lexicography.
1. As the same applies to the lemmatization of word forms, we train
   spaCy's probabilistic lemmatizer on a large dataset of example
   sentences from ZDL's own lexical resources. These sentences have
   been lemmatized with the deterministic lemmatizer
   [DWDSmor](https://github.com/zentrum-lexikographie/dwdsmor) and
   therefore adhere to the expected conventions of the ZDL.

Evaluated against test splits of the aforementioned datasets, the two
models trained perform as follows:

| Annotation Type           | Accuracy (static emb.) | Accuracy (contextual emb.) |
|:--------------------------|-----------------------:|---------------------------:|
| PoS Tagging               |                 97.69% |                     98.45% |
| Morphological Features    |                 91.33% |                     93.97% |
| Syntactic Relations (LAS) |                 92.45% |                     95.52% |
| Syntactic Relations (UAS) |                 94.69% |                     96.77% |
| Lemmatization             |                 98.62% |                     98.64% |
| Named Entities (f-score)  |                 75.19% |                     87.71% |

One model is trained on static embeddings and should provide higher
throughput on CPU architectures, while the other is trained on
contextual embeddings provided by a transformer base model and should
provide higher accuracy but require GPU hardware for annotating larger
corpora.

## Usage

The models are available from our package registry at [Git.UP](https://gitup.uni-potsdam.de/):

``` shell
pip install de-zdl-lg --index-url https://gitup.uni-potsdam.de/api/v4/projects/21461/packages/pypi/simple
pip install de-zdl-dist --index-url https://gitup.uni-potsdam.de/api/v4/projects/21461/packages/pypi/simple
```

The first package (with suffix `-lg`) contains the model with static
word embeddings, the second (with suffix `-dist`) provides the model
based on [DistilBERT](https://arxiv.org/abs/1910.01108).

Once installed, you can use the pipelines like any other spaCy pipeline, e.g.

``` python
>>> import spacy
>>> nlp = spacy.load("de_zdl_lg") # or "de_zdl_dist"
>>> [(e, e.label_) for e in nlp("Heiner Müller wurde am 9. Januar 1929 in Eppendorf in Sachsen geboren.").ents]
[(Heiner Müller, 'PER'), (Eppendorf, 'LOC'), (Sachsen, 'LOC')]
```

## Training Datasets


* Benikova, Darina, Chris Biemann, und Marc Reznicek. „NoSta-D Named Entity Annotation for German: Guidelines and Dataset“. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), herausgegeben von Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, und Stelios Piperidis, 2524–31. Reykjavik, Iceland: European Language Resources Association (ELRA), 2014. https://aclanthology.org/L14-1251/.
* Berlin-Brandenburg Academy of Sciences and Humanities (BBAW) (ed.) (n.d.). DWDS – Digitales Wörterbuch der deutschen Sprache: Das Wortauskunftssystem zur deutschen Sprache in Geschichte und Gegenwart. https://www.dwds.de/
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
