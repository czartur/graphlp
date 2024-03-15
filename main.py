# cli entry point

import argparse
from dataclasses import dataclass, fields
from typing import List, Literal, Union
import matplotlib.pyplot as plt
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
    model: Union[Literal["NLP"], Literal["DGP"], Literal["ISO"]] = "NLP"
    sample_size: int = 3
    # model parameters
    nlp_initial_embedding_from: Literal["DGP"] = "DGP"
    nlp_solver: str = "ipopt"
    dgp_kdim: int = 3
    dgp_solver: str = "cplex"
    dgp_projection: Union[Literal["pca"], Literal["barvinok"]] = "pca"


def dataclass_to_argparse(dc):
    parser = argparse.ArgumentParser()
    for dc_field in fields(dc):
        field_type = dc_field.type
        field_name = dc_field.name.replace('_', '-')
        if field_type is bool:
            parser.add_argument(
                f'--{field_name}',
                action='store_true',
                help=f'{field_name} (default: {dc_field.default})'
            )
            parser.add_argument(
                f'--no-{field_name}',
                dest=field_name,
                action='store_false'
            )
            parser.set_defaults(**{field_name: dc_field.default})
        elif field_type is int:
            parser.add_argument(
                f'--{field_name}',
                type=int,
                default=dc_field.default,
                help=f'{field_name} (default: {dc_field.default})'
            )
        else:
            parser.add_argument(
                f'--{field_name}',
                default=dc_field.default,
                help=f'{field_name} (default: {dc_field.default})'
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

    print("Enriching the graph-of-words.")
    if config.similarity == "path_word":
        graph.enrich(path_word_similarity)
    adjacency_matrix = graph.adjacency_matrix()

    print("Running the model.")
    if config.model == "DGP":
        dgp = embedding_model.DGP(
            'models/dgp_ddp.mod',
            Kdim=config.dgp_kdim,
            solver=config.dgp_solver,
            projection=config.dgp_projection,
        )
        embedding = dgp.embed(adjacency_matrix)
    elif config.model == "NLP":
        if config.nlp_initial_embedding_from == "DGP":
            dgp = embedding_model.DGP(
                'models/dgp_ddp.mod',
                Kdim=config.dgp_kdim,
                solver=config.dgp_solver,
                projection=config.dgp_projection,
            )
            enriched_embeddings = dgp.embed(adjacency_matrix)
        else:
            print("Initial embeddings method not allowed.")
            return

        nlp = embedding_model.NLP(
            'models/dgp.mod',
            enriched_embeddings,
            solver=config.nlp_solver
        )
        embedding = nlp.embed(adjacency_matrix)
    elif config.model == "ISO":
        iso = embedding_model.IsometricEmbedding()
        embedding = iso.embed(adjacency_matrix)
    else:
        print("Model not supported")
        return

    print("Visualizing the embeddings.")

    all_words = " ".join(corpus).split(" ")
    words: List[str] = np.random.choice(
        all_words, config.sample_size).tolist()

    visualize_embeddings(embedding, words, graph.get_word_idx)

    while True:
        if plt.waitforbuttonpress(0):
            plt.close()
            user_input = input(
                "Enter 'n' to input a new word, "
                + "'q' to quit or "
                + "'r' to a new random word: ")
            if user_input == 'q':
                break
            elif user_input == 'r':
                new_word = np.random.choice(
                    [w for w in all_words if w not in words])
                words.append(new_word)
                visualize_embeddings(embedding, words, graph.get_word_idx)
            elif user_input == 'n':
                new_word = input("Enter a word you want to visualize: ")
                if graph.get_word_idx(new_word) == 0:
                    print("Word not in vocabulary.")
                else:
                    words.append(new_word)
                visualize_embeddings(embedding, words, graph.get_word_idx)

            else:
                print("Input not recognized. Closing.")
                break


if __name__ == "__main__":
    main()
