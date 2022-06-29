<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: DWDS NLP Pipeline

This project lets you train a part-of-speech tagger, morphologizer, lemmatizer and dependency parser from a DWDS corpus.  It takes care of data preparation, converting it to spaCy's format and training and evaluating the model. Note that multi-word tokens will be merged together when the corpus is converted since spaCy does not support multi-word token expansion.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `preprocess` | Convert the data to spaCy's format |
| `train` | Train ud_de_dwds |
| `train-trf` | Train ud_de_dwds transformer |
| `evaluate` | Evaluate on the test data and save the metrics |
| `evaluate-trf` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `package-trf` | Package the trained model so it can be installed |
| `clean` | Remove intermediate files |
| `visualize-model` | Visualize the model's output interactively using Streamlit |
| `visualize-model-trf` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` &rarr; `package` |
| `all-trf` | `preprocess` &rarr; `train-trf` &rarr; `evaluate-trf` &rarr; `package-trf` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
