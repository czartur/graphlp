__all__ = ["GraphOfWords"]
from typing import Callable, List
import networkx as nx


class GraphOfWords:
    def __init__(self, corpus: List[str], radius: int = 3) -> None:
        ...

    def enrich(self, similarity: Callable[[str, str], float]) -> None:
        ...

    def get_word_idx(self, word: str) -> int:
        ...

    @property
    def graph(self) -> nx.Graph:
        ...
