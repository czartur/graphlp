import argparse
import pickle
import os
from dataclasses import dataclass, fields
from typing import List, Literal, Union
import matplotlib.pyplot as plt
import numpy as np
import nltk.corpus as corpi
import nltk

from graphlp.graph_of_words import GraphOfWords
from graphlp import embedding_model, similarity as sim
from graphlp.visualize import visualize_embeddings
from graphlp.metrics import print_error_summary


@dataclass
class Configuration:
    radius: int = 3
    corpus: Union[Literal["brown"], Literal["wordnet"]] = "brown"
    training_size: int = 100
    similarity: Union[Literal["path"], Literal["wup"], None] = "path"
    model: Union[Literal["NLP"], Literal["DDP"], Literal["ISO"]] = "NLP"
    sample_size: int = 3
    save_path: str = ""
    load_path: str = ""
    # model parameters
    nlp_initial_embedding_from: Union[Literal["DDP"], Literal["ISO"]] = "DDP"
    nlp_solver: str = "ipopt"
    ddp_kdim: int = 3
    ddp_solver: str = "cplex"
    ddp_projection: Union[Literal["pca"], Literal["barvinok"]] = "pca"
    plot: bool = True

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


def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    config = parse_args_to_dataclass(Configuration)

    if config.load_path:
        if os.path.exists(config.load_path):
            print(f"Loading model and embeddings from {config.load_path}")
            saved_data = load_data(config.load_path)
            graph = saved_data['graph']
            embedding = saved_data['embedding']
        else:
            print(f"No file found at {config.load_path}. Aborting execution.")
            return
    else:
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
        if config.similarity == "path":
            graph.enrich(sim.path_word_similarity)
        elif config.similarity == "wup":
            graph.enrich(sim.wup_word_similarity)
        else:
            print("Similaity measure not supported.")
            return
        adjacency_matrix = graph.adjacency_matrix()

        print("Running the model.")
        if config.model == "DDP":
            ddp = embedding_model.DDP(
                'models/dgp_ddp.mod',
                Kdim=config.ddp_kdim,
                solver=config.ddp_solver,
                projection=config.ddp_projection,
            )
            embedding = ddp.embed(adjacency_matrix)

        elif config.model == "NLP":
            if config.nlp_initial_embedding_from == "DDP":
                ddp = embedding_model.DDP(
                    'models/dgp_ddp.mod',
                    Kdim=config.ddp_kdim,
                    solver=config.ddp_solver,
                    projection=config.ddp_projection,
                )
                enriched_embeddings = ddp.embed(adjacency_matrix)
            elif config.nlp_initial_embedding_from == "ISO":
                iso = embedding_model.IsometricEmbedding()
                enriched_embeddings = iso.embed(adjacency_matrix)
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

    if config.save_path:
        print(f"Saving model and embeddings to {config.save_path}")
        save_data({'graph': graph, 'embedding': embedding, 'config': config}, config.save_path)

    print("Evaluating error statistics.")
    print_error_summary(embedding, graph.adjacency_matrix())

        
    if config.plot:
        print("Visualizing the embeddings.")

        all_words = graph.all_words
        words: List[str] = np.random.choice(
        all_words, config.sample_size).tolist()

        print(words)
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
