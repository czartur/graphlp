# GraphLP

This project aims to understand, implement and evaluate mathematical programming approaches to generate graph embeddings on natural language texts.

This was done under the INF580 course at Ecole Polytechnique.

## Installing
The code is distributed as a python library. To be able to build it you need to have setuptools installed (it usually comes with default python distributions).

To install the necessary dependencies, install the library via PIP:
```bash
pip install .
```

If you want to edit the code in the library and see changes directly, install it in editable mode:
```bash
pip install -e .
```

After installation, you should be able to use it as usual:
```python
import graphlp
from graphlp.similarity import path_word_similarity
print(path_word_similarity('cat', 'dog'))

from graphlp.graph_of_words import GraphOfWords
graph = GraphOfWords(corpus=['My name is John', 'Mine is Kevin!'], radius=3)
```

## CLI
On top of the library, we distribute a CLI tool that helps you easily run different models and visualize your word embeddings on a 3D space. Note that to run this script you should have the library installed.

For help on how to use it:
```bash
python main.py -h
```

A sample usage is the following
```bash
python main.py \
    --radius 4 \            ## radius for the GoW
    --corpus brown \        ## NLTK corpus for training
    --training-size 10 \    ## number of sentences to consider from the corpus
    --similarity wup \      ## The similarity used to enrich the GoW. options: [wup, path, lch]
    --model nlp \           ## embedding model. options [NLP, DGP, ISO]
    --sample-size 5 \       ## number of words to display on the visualization tool
    --save-path model.pkl \ ## where to save the model once you're done
    --load-path model.pkl \ ## where to load a pretrained one from. Will skip all training.

    ## if using NLP as the model
    --nlp-initial-embedding-from DGP \  ## initial embeddings for NLP run
    --nlp-solver ipopt \    ## solver to be used for NLP

    ## if using DGP as the model (or as NLP initializer)
    --dgp-kdim 3 \          ## dimension of the embedding
    --dgp-solver cplex \    ## solver to be used for DGP
    --dgp-projection pca    ## projection method. options [pca, barvinok]
```

This will run the desired model on the subset of the corpus to generate embeddings. After those are generated, a visualization tool will be ran to see your embeddings in a 3D space!

Uppon pressing any key you will leave the tool and be prompted on the terminal for the next step.
We currently have support for adding a word of your choice or a random word and relauching the visualization.

