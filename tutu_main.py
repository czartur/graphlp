import os
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk

def construct_graph(
    corpus: list[str],
    window_size: int = 3,
    plot: bool = False,
) -> nx.Graph:
    ## Pre processing 
    # tokenize ignoring stop words
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus) # sparse matrix (n_docs x n_tokens)

    # vocabulary=vectorizer.vocabulary_ # token --> idx

    # tokens and scores (as being the sum of the tfidf in each sample)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, np.ravel(X.sum(axis=0)))) 

    ## Graph of words
    # count co-occurences in each window
    co_occurrences = defaultdict(int)
    for i,text in enumerate(corpus):
        # print(i)
        words = [word for word in text.lower().split() if word in feature_names]
        for i in range(len(words)):
            for j in range(i+1, min(i+window_size, len(words))):
                pair = tuple(sorted([words[i], words[j]]))
                co_occurrences[pair] += 1

    # build the graph 
    G = nx.Graph()
    for (word1, word2), count in co_occurrences.items():
        weight = (tfidf_scores[word1] + tfidf_scores[word2]) * count
        G.add_edge(word1, word2, weight=weight)

    # some viz
    if plot:
        pos = nx.spring_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        nx.draw(G, pos, node_color="#A0CBE2", edge_cmap=plt.cm.Blues, edgelist=edges, edge_color=weights, width=1.0, with_labels=True) # colors are weight based
        plt.show()
     
    return G

def write_dat(
    G: nx.Graph,
    path: str,
    embed_dim: int = 3,
):  
    n = len(G.nodes())
    node_mapping = {node: idx for idx,node in enumerate(G.nodes(),1)}
    iG = nx.relabel_nodes(G, node_mapping)

    with open(path, "w") as file: 
        file.write(f"param Kdim := {embed_dim};\n")
        file.write(f"param n := {n};\n")
        file.write(f"param : E : c I :=\n")
        for u,v,w in iG.edges.data('weight'):
            if u == v: continue
            file.write(f"{u} {v} {w} 1\n")
            file.write(f"{v} {u} {w} 1\n")
        file.write(";")

def read_dat(G: nx.Graph, path: str, embed_dim: int = 3):
    n = len(G.nodes)
    K = embed_dim
    with open(path, "r") as file:
        # embedding matrix
        X = np.zeros(shape=(n,K))
        for line in file.readlines()[2:-1]:
            i,j,val = line.strip().split()
            i,j,val = int(i)-1, int(j)-1, float(val)
            X[i,j] = val
        embedding = {node: X[idx, :] for idx, node in enumerate(G.nodes())}
    
    return embedding 

def plot_ngrams(
    G,
    embedding,
    n_plot: int = 3,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vocabulary_inv = {idx: token for idx,token in enumerate(G.nodes(),1)}
    
    
    sample = np.random.choice(len(embedding), n_plot)
    sents, embs = zip(*(list(embedding.items())[i] for i in sample))

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

def most_similar(
    G: nx.Graph, 
    embedding: dict[str, np.ndarray],
    word: str,
) -> list[tuple[str, float]]:
    
    emb_word = embedding[word]
    distance = [(node,np.linalg.norm(emb_word - emb)) for node,emb in embedding.items() if node != word]
    return sorted(distance, key=lambda x:x[1])

if __name__ == "__main__":
    # sample text
    # corpus = ["This is a simple example sentence demonstrating a graph of words.",
    #          "Another sentence with words to demonstrate the graph constructio."]
    
    from nltk.corpus import brown
    nltk.download('brown')
    corpus = [' '.join(words) for words in brown.sents()]

    G = construct_graph(corpus[:500], plot=False)
    print(f"n_nodes={len(G.nodes())}\nn_edges={len(G.edges())}")

    write_dat(G, "test.dat", 16) 

    script = "dgp_ddp.py"
    # script = "dgp_dualddp.py"
    # script = "dgp_ddp.py"
    os.system(f'python3 {script} test.dat noplot')

    embs = read_dat(G, "test-sol.dat", 3) 
    plot_ngrams(G, embs, n_plot=5)
    
    import random
    word = random.sample(list(G.nodes()),1)[0]
    close = most_similar(G, embs, word) 
    print(word)
    print(close[:5])
