__all__ = ["GraphOfWords"]

import networkx as nx
import numpy as np
from nltk.tokenize import wordpunct_tokenize
# from itertools import product
from tqdm.contrib.itertools import product
from typing import Callable, Dict, List


class GraphOfWords:
    def __init__(self, corpus: List[str], radius: int = 3,
                 tokenize: Callable[[str], List[str]] = wordpunct_tokenize
                 ) -> None:
        self._corpus = corpus
        self._radius = radius
        self._vocabulary: Dict[str, int] = self._create_vocabulary(
            corpus, tokenize)
        tokens = [
            [self._vocabulary[word] for word in tokenize(sentence)]
            for sentence in corpus
        ]
        self._graph = self._create_graph(tokens, radius)
        self.enrich(lambda _, __: 1.)

    def enrich(self, similarity: Callable[[str, str], float]) -> None:
        attributes = {}
        for (w1, i1), (w2, i2) in product(
                items := self._vocabulary.items(), items):
            sim = similarity(w1, w2)
            if (i1, i2) in self._graph.edges:
                attributes[(i1, i2)] = {"similarity": sim}
            if (i2, i1) in self._graph.edges:
                attributes[(i1, i2)] = {"similarity": sim}
        nx.set_edge_attributes(self._graph, attributes)

    def get_word_idx(self, word: str) -> int:
        return self._vocabulary.get(word, 0)

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def adjacency_matrix(self) -> np.ndarray:
        n = len(self._vocabulary.values())
        adjacency_matrix = np.zeros((n, n))
        for u, v, data in self.graph.edges(data=True):
            similarity = data.get("similarity", 0)
            adjacency_matrix[u][v] = similarity
            adjacency_matrix[v][u] = similarity
        return adjacency_matrix

    @staticmethod
    def _create_vocabulary(
        corpus: List[str],
        tokenize: Callable[[str], List[str]]
    ) -> Dict[str, int]:
        tokens = [word for sentence in corpus for word in tokenize(sentence)]
        special_tokens = ["<oov>", "<start>", "<end>"]
        unique_tokens = special_tokens + \
            list(set(tokens) - set(special_tokens))
        return {token: index for index, token in enumerate(unique_tokens)}

    @staticmethod
    def _create_graph(
        tokens: List[List[int]],
        radius: int,
    ) -> nx.Graph:
        edges = [
            (w1, w2)
            for r in range(1, radius+1)
            for token_sentence in tokens
            for w1, w2 in zip(token_sentence, token_sentence[r:])
        ]
        graph = nx.Graph()
        graph.add_edges_from(edges, directed=False)
        return graph
