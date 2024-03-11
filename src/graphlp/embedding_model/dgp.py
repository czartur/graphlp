__all__ = ["DGP"]

from amplpy import AMPL
import numpy as np

from graphlp.embedding_model.abstract import EmbeddingModel
from graphlp.embedding_model.ampl_parsers import ampl_to_numpy, numpy_to_ampl


class DGP(EmbeddingModel):
    def __init__(self, model_path: str, solver: str = 'cplex', Kdim: int = 3):
        self.model = AMPL()
        self.model.read(model_path)
        self.model.setOption('solver', solver)
        self._kdim = Kdim

    def embed(self, graph: np.ndarray) -> np.ndarray:
        assert np.allclose(graph, graph.T)

        n = len(graph)
        ampl_graph = numpy_to_ampl(graph)

        # set variables
        self.model.param['n'] = n
        self.model.set['E'] = list(ampl_graph.keys())
        self.model.param['c'] = ampl_graph

        # solve
        self.model.solve()

        # get dot product matrix
        ampl_X = self.model.getVariable('X')
        X = ampl_to_numpy(ampl_X, shape=(n, n))

        # retrieve realizations
        return self._pca(X, self._kdim)

    @staticmethod
    def _pca(X: np.ndarray, Kdim: int) -> np.ndarray:
        assert Kdim <= len(X)

        # X ~ L @ L.T
        # L = vecs @ sqrt(max(vals,0)) [nearest PSD]
        vals, vecs = np.linalg.eigh(X)
        x = vecs @ np.diag(np.sqrt(np.maximum(vals, 0)))

        # filter dims corresponding to largest eigenvalues
        x = x[:, -Kdim:]

        return x
