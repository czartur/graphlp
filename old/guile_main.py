from collections import defaultdict
from typing import Callable
from amplpy import AMPL
import numpy as np

from nltk.tokenize import wordpunct_tokenize


corpus: list[str]
Vocabulary = dict[str, int]
Ngram = tuple[int, ...]
OccuranceGraph = dict[tuple[Ngram, Ngram], int]
DistanceGraph = dict[tuple[Ngram, Ngram], float]


def ngramize(array: list[int], n: int) -> list[Ngram]:
    return [tuple(array[i:i+n])
            for i in range(len(array) - n + 1)]


def tokenize(sentence: str) -> list[str]:
    return ["<start>", *wordpunct_tokenize(sentence), "<end>"]


def create_vocabulary(corpus: list[str]) -> Vocabulary:
    tokens = [word for sentence in corpus for word in tokenize(sentence)]
    special_tokens = ["<oov>", "<start>", "<end>"]
    unique_tokens = special_tokens + list(set(tokens) - set(special_tokens))
    return {token: index for index, token in enumerate(unique_tokens)}


def pre_process(corpus: list[str]) -> tuple[Vocabulary, list[list[int]]]:
    vocabulary = create_vocabulary(corpus)
    tokens = [[vocabulary[word]
               for word in tokenize(sentence)]for sentence in corpus]
    return vocabulary, tokens


def create_occurance_graph(
    sentence: list[int],
    context_len: int = 2
) -> OccuranceGraph:
    ngrammed = ngramize(sentence, context_len)
    graph = defaultdict(lambda: 0)
    for edge in zip(ngrammed, ngrammed[1:]):
        graph[edge] += 1
    return graph


def normalize_occurance_graph(graph: OccuranceGraph) -> DistanceGraph:
    neighbors = defaultdict(list)
    for i, j in graph.keys():
        neighbors[i].append(j)
    new_graph = {}
    for node, neighs in neighbors.items():
        total = sum(graph[(node, neigh)] for neigh in neighs) 
        for neigh in neighs:
            new_graph[(node, neigh)] = graph[(node, neigh)] / total
    return new_graph


def enrich_graph(
    graph: DistanceGraph,
    simmilarity: Callable[[int, int], float]
) -> DistanceGraph:
    ...


# def embed(
#     graph: DistanceGraph,
#     embed_dim: int = 16
# ) -> dict[Ngram, list[float]]:
#
#     nodes = list(
#         set(node for node, _ in graph.keys()).union(
#             set(node for _, node in graph.keys()))
#     )
#     node_hash = {node: idx+1 for idx, node in enumerate(nodes)}
#     node_hash_inv = {idx+1: node for idx, node in enumerate(nodes)}
#     graph_hash = {
#         (node_hash[i], node_hash[j]): val for (i, j), val in graph.items()}
#
#     ampl = AMPL()
#     ampl.eval(r"""
#         param n integer, > 0;
#         set V := {1..n};
#         set E within {V, V};
#         param d{V, V};
#         param dim integer, > 0;
#         set K := {1..dim};
#
#         var x{V, K} default Uniform(-10, 10);
#
#         minimize slack:
#             sum{(u, v) in E}
#             (
#                 (sum{k in K} (x[u, k] - x[v, k]) ^ 2)
#                 - (d[u, v] ^ 2)
#             )^2
#         ;
#               """)
#     ampl.param["n"] = len(nodes)
#     ampl.param["dim"] = embed_dim
#     ampl.set["E"] = graph_hash.keys()
#     ampl.param["d"] = graph_hash
#     ampl.solve(solver="knitro")
#     embedding = ampl.get_variable("x").get_values().to_dict()
#
#     emb: dict[Ngram, list[float]] = defaultdict(lambda: [0]*embed_dim)
#     for i in range(1, len(nodes) + 1):
#         for j in range(1, embed_dim+1):
#             emb[node_hash_inv[i]][j-1] = embedding[(i, j)]

def write_dat(
    path: str,
    graph: DistanceGraph,
    embed_dim: int = 16,
): 
    nodes = list(
        set(node for node, _ in graph.keys()).union(
            set(node for _, node in graph.keys()))
    )
    node_hash = {node: idx+1 for idx, node in enumerate(nodes)}
    # node_hash_inv = {idx+1: node for idx, node in enumerate(nodes)}

    graph_hash = {
        (node_hash[i], node_hash[j]): val for (i, j), val in graph.items()}

    file = open(path, "w")
    file.write(f"param Kdim := {embed_dim};\n")
    file.write(f"param n := {len(nodes)};\n")
    file.write(f"param : E : c I :=\n")
    for (i, j), w in graph_hash.items():
        file.write(f"{i} {j} {w} 1\n")
    file.write(";")

def read_dat(
    path: str,
    graph: DistanceGraph,
    embed_dim: int = 16,
) -> dict[Ngram, np.ndarray]: 
    
    nodes = list(
        set(node for node, _ in graph.keys()).union(
            set(node for _, node in graph.keys()))
    )
    node_hash = {node: idx+1 for idx, node in enumerate(nodes)}
    # node_hash_inv = {idx+1: node for idx, node in enumerate(nodes)}

    graph_hash = {
        (node_hash[i], node_hash[j]): val for (i, j), val in graph.items()}

    file = open(path, "r")
    
    # embedding matrix
    X = np.zeros((len(nodes), embed_dim))
    for line in file.readlines()[2:-1]:
        n, k, val = line.strip().split()
        X[int(n)-1,int(k)-1] = val

    embedding = {node: X[idx-1, :] for node, idx in node_hash.items()}
    return embedding

def sentence_similarity(
    s1: str,
    s2: str,
    n: int,
    vocabulary: Vocabulary,
    embedding: dict[Ngram, list[float]]
) -> float:
    t1 = [vocabulary.get(t, 0) for t in tokenize(s1)]
    t2 = [vocabulary.get(t, 0) for t in tokenize(s2)]

    ng1 = ngramize(t1, n)
    ng2 = ngramize(t2, n)
    
    print(ng1)
    e1 = np.array([embedding[ng] for ng in ng1])

    e2 = np.array([embedding[ng] for ng in ng2])

    emb1 = e1.mean(axis=0)
    nemb1 = np.linalg.norm(emb1)
    emb1 = np.where(nemb1 == 0, emb1, emb1/nemb1)
    emb2 = e2.mean(axis=0)
    nemb2 = np.linalg.norm(emb2)
    emb2 = np.where(nemb2 == 0, emb2, emb2/nemb2)
    print(emb1, emb2)

    return np.dot(emb1, emb2)

def plot_ngrams(
    vocabulary: Vocabulary,
    embedding: dict[Ngram, np.ndarray],
    n_plot: int = 3,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vocabulary_inv = {idx: token for token,idx in vocabulary.items()}
    
    keys = list(embedding.keys())
    idx_sample = np.random.choice(len(keys), n_plot)
    embs = [embedding[keys[idx]] for idx in idx_sample]
    sents = [vocabulary_inv[keys[idx][0]] + " " + vocabulary_inv[keys[idx][1]] for idx in idx_sample]

    #Extracting individual components for plotting
    xs = [v[0] for v in embs]
    ys = [v[1] for v in embs]
    zs = [v[2] for v in embs]

    # Creating the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the vectors
    scatter = ax.scatter(xs, ys, zs)

    # Adding labels
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # Looping through each vector to add its sentence as a label
    for i, sentence in enumerate(sents):
        ax.text(xs[i], ys[i], zs[i], sentence, size=10, zorder=1, color='k')

    # Display the plot
    plt.show()

if __name__ == "__main__":
    import nltk
    nltk.download("gutenberg")
    from nltk.corpus import gutenberg
    csize = 1
    maxsize = 10000
    n = 2
    corpus: list[str] = [gutenberg.raw(idx)[:maxsize]
                         for idx in gutenberg.fileids()[:csize]]
    # corpus = [
    #     "hey my name love bread",
    #     "my name love"
    # ]
    # with open("corpus.txt", "w") as file:
        # file.write(''.join(corpus))
    
    voc, tokcor = pre_process(corpus)
    g = create_occurance_graph([t for sentence in tokcor for t in sentence], n)
    gg = normalize_occurance_graph(g)
    write_dat("test.dat", gg, 16)
    
    import os
    os.chdir('solutions')
    os.system('python3 dgp_ddp.py ../test.dat noplot')
    os.chdir('..')
    embs = read_dat("solutions/test-sol.dat", gg, 3)
    
    plot_ngrams(voc, embs, 3) 
    # embs = embed(gg)
    # s1 = "and the shadow of"
    # s2 = "or playful is dearly"
    # s2 = "the evening nursed the man"
#
    # sim = sentence_similarity(s1, s2, n, voc, embs)
    # print(sim)
