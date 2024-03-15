__all__ = ["path_word_similarity",
           "wup_word_similarity", "lch_word_similarity"]
from nltk.corpus import wordnet as wn


def _nltk_word_similarity(
        word1,
        word2,
        comparator
):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if not synsets1 or not synsets2:
        return 0.0  # If no synsets are found for either word

    max_similarity = 0.0
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 is None or synset2 is None:
                continue
            similarity = comparator(synset1, synset2)
            if similarity and similarity > max_similarity:
                max_similarity = similarity

    return max_similarity


def path_word_similarity(word1, word2):
    return _nltk_word_similarity(
        word1, word2,
        lambda x, y: x.path_similarity(y))


def wup_word_similarity(word1, word2):
    return _nltk_word_similarity(
        word1, word2,
        lambda x, y: x.wup_similarity(y))


def lch_word_similarity(word1, word2):
    return _nltk_word_similarity(
        word1, word2,
        lambda x, y: x.lch_similarity(y))
