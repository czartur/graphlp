__all__ = ["NLP"]

from graphlp.embedding_model.abstract import EmbeddingModel
from graphlp.embedding_model.ampl_parsers import numpy_to_ampl, ampl_to_numpy
from amplpy import AMPL
import numpy as np


class NLP(EmbeddingModel):
    def __init__(
        self,
        model_path: str,
        initial_embedding: np.ndarray,
        solver: str = 'ipopt'
    ):
        self.model = AMPL()
        self.model.read(model_path)
        self.model.setOption('solver', solver)
        self._initial_embedding = initial_embedding

    def embed(self, graph: np.ndarray) -> np.ndarray:
        assert np.allclose(graph, graph.T)

        n, Kdim = self._initial_embedding.shape

        ampl_graph = numpy_to_ampl(graph)
        print(self._initial_embedding)
        ampl_intial_embedding = numpy_to_ampl(self._initial_embedding)
        print(ampl_intial_embedding)

        # set variables
        self.model.param['Kdim'] = Kdim
        self.model.param['n'] = n
        self.model.set['E'] = list(ampl_graph.keys())
        self.model.param['c'] = ampl_graph
        self.model.var['x'] = ampl_intial_embedding

        # solve
        self.model.solve()

        # get realizations
        ampl_x = self.model.var['x']
        x = ampl_to_numpy(ampl_x, shape=(n, Kdim))

        return x
