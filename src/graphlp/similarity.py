from nltk.corpus import wordnet as wn


def path_word_similarity(word1, word2):
    """
    Calculate similarity between two words using NLTK and WordNet.

    Parameters:
    word1 (str): The first word.
    word2 (str): The second word.

    Returns:
    similarity (float): The similarity score between the two words (0 to 1).
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if not synsets1 or not synsets2:
        return 0.0  # If no synsets are found for either word

    max_similarity = 0.0

    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity and similarity > max_similarity:
                max_similarity = similarity

    return max_similarity

