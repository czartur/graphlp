# cli entry point

import argparse
from dataclasses import dataclass, fields
from typing import Literal, Union
import numpy as np
import nltk.corpus as corpi
import nltk


from graphlp.graph_of_words import GraphOfWords
from graphlp.similarity import path_word_similarity
from graphlp import embedding_model
from graphlp.visualize import visualize_embeddings


@dataclass
class Configuration:
    radius: int = 3
    corpus: Union[Literal["brown"], Literal["wordnet"]] = "brown"
    training_size: int = 100
    similarity: Union[Literal["path_word"], None] = "path_word"
    model: Union[Literal["NLP"], Literal["DGP"], Literal["UIE"]] = "NLP"
    words: str = "hello, hi"
    # model parameters
    nlp_initial_embedding_from: Literal["DGP"] = "DGP"
    nlp_solver: str = "ipopt"
    dgp_kdim: int = 3
    dgp_solver: str = "cplex"


def dataclass_to_argparse(dc):
    parser = argparse.ArgumentParser()
    for dc_field in fields(dc):
        field_type = dc_field.type
        if field_type is bool:
            parser.add_argument(
                f'--{dc_field.name}',
                action='store_true',
                help=f'{dc_field.name} (default: {dc_field.default})'
            )
            parser.add_argument(
                f'--no-{dc_field.name}',
                dest=dc_field.name,
                action='store_false'
            )
            parser.set_defaults(**{dc_field.name: dc_field.default})
        elif field_type is int:
            parser.add_argument(
                f'--{dc_field.name}',
                type=int,
                default=dc_field.default,
                help=f'{dc_field.name} (default: {dc_field.default})'
            )
        else:
            parser.add_argument(
                f'--{dc_field.name}',
                default=dc_field.default,
                help=f'{dc_field.name} (default: {dc_field.default})'
            )
    return parser


def parse_args_to_dataclass(dc_cls):
    parser = dataclass_to_argparse(dc_cls)
    args = parser.parse_args()
    return dc_cls(**vars(args))


def main():

    config = parse_args_to_dataclass(Configuration)

    print("Downloading selected corpus.")
    nltk.download(config.corpus)
    corpus_loader = getattr(corpi, config.corpus)
    if corpus_loader is None:
        return
    corpus = [
        " ".join(sentence)
        for sentence in corpus_loader.sents()[:config.training_size]
    ]

    print("Creating the graph-of-words.")
    graph = GraphOfWords(corpus, radius=config.radius)
    adjacency_matrix = graph.adjacency_matrix()
    embedding = np.zeros(0)

    print("Running the model.")
    if config.similarity == "path_word":
        graph.enrich(path_word_similarity)
    if config.model == "DGP":
        dgp = embedding_model.DGP(
            'old/dgp_ddp.mod',
            Kdim=config.dgp_kdim,
            solver=config.dgp_solver
        )
        embedding = dgp.embed(adjacency_matrix)
    elif config.model == "NLP":
        if config.nlp_initial_embedding_from == "DGP":
            dgp = embedding_model.DGP(
                'old/dgp_ddp.mod',
                Kdim=config.dgp_kdim,
                solver=config.dgp_solver
            )
            enriched_embeddings = dgp.embed(adjacency_matrix)
            print(enriched_embeddings)
        else:
            print("Initial embeddings method not allowed.")
            return

        nlp = embedding_model.NLP(
            'old/dgp.mod',
            enriched_embeddings,
            solver=config.nlp_solver
        )
        embedding = nlp.embed(adjacency_matrix)
    elif config.model == "UIE":
        uie = embedding_model.UIE()
        embedding = uie.embed(adjacency_matrix)
    else:
        print("Model not supported")
        return

    print("Visualizing the embeddings.")
    words = config.words.split(",")
    visualize_embeddings(embedding, words, graph.get_word_idx)


if __name__ == "__main__":
    main()
