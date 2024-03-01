from graphlp import graph_of_words
from graphlp import similarity

c = [
        "hi, my name is gui.",
        "hello my name is artur"
    ]
g = graph_of_words.GraphOfWords(c, 3)
g.graph.edges
g.graph.nodes
g._vocabulary
g.get_word_idx("artur")
g.get_word_idx("name")
g.get_word_idx("is")

similarity.word_similarity("cat", "car")
# g.enrich(similarity.word_similarity)

g.adjacency_matrix()
