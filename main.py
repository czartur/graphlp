from graphlp import graph_of_words
from graphlp import similarity
from graphlp.embedding_model.dgp import DGP
from graphlp.embedding_model.nlp import NLP
from graphlp import visualize
c = [
        "hi, my name is gui.",
        "hi my name is artur"
    ]
g = graph_of_words.GraphOfWords(c, 3)
g.graph.edges
g.graph.nodes
g._vocabulary
g.get_word_idx("artur")
g.get_word_idx("name")
g.get_word_idx("is")

# similarity.word_similarity("cat", "car")
# g.enrich(similarity.word_similarity)

graph = g.adjacency_matrix()
dgp = DGP('old/dgp_ddp.mod')
nlp = NLP('old/dgp.mod')

embedding = dgp.embed(graph, Kdim=3)
embedding_refined = nlp.embed(graph, embedding)
visualize.visualize_embeddings(embedding_refined, ['gui', 'hello'], g.get_word_idx)
