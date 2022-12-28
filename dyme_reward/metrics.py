# Code adapted from https://github.com/florianvonunold/DYME
"""
This file implements different metric calculation functions for a given conversation.
"""
# Note: some imports are defined inside the methods of the metrics (for cases where only some metrics are computed)
import os
import string
from pathlib import Path

import numpy as np
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from metric_helpers import cosine_similarity, _get_sentiment_multiplier
from config import models_dir

stopwords = stopwords.words('english')
question_words = {'who', 'what', 'why', 'where', 'how', 'when'}
_ = [stopwords.remove(q) for q in question_words]
punct = list(string.punctuation)
contractions = ["'s", "'d", "'ld", "n't", "'re", "'ll", "'ve"]
filters = set(stopwords + contractions + punct)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def question(conversation):
    """Counts whether each utterance in the given conversation contains a question (yes: 1, no: 0)"""
    num_turns = len(conversation)
    is_question_in_utterance = np.zeros(num_turns)

    for i, utterance in enumerate(conversation):
        if any(question_word in utterance for question_word in question_words) and '?' in utterance:
            is_question_in_utterance[i] = 1

    return is_question_in_utterance


def conversation_repetition(conversation):
    """Counts the number of distinct words in the current utterance that were also in any of the previous utterances"""
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    filtered = [set(utterance).difference(filters) for utterance in conversation]  # filter stopwords, contractions and punctuation

    for i in range(1, num_turns):
        current = filtered[i]
        prev = set.union(*filtered[:i])
        repeats = current.intersection(prev)
        num_repeats_in_utterances[i] = len(repeats)

    return num_repeats_in_utterances


def self_repetition(conversation):  # called 'reward_conversation_repetition' (for bot utterances) in original paper
    """
    Counts the number of distinct words in the current utterance that were also in any of the previous utterances of
    the current speaker (assuming two-speaker multi-turn dialog)
    """
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    filtered = [set(utterance).difference(filters) for utterance in conversation]  # filter stopwords, contractions and punctuation

    # first and second utterance can't repeat any word of previous utterances of the current speaker
    num_repeats_in_utterances[0] = 0
    if num_turns > 1:
        num_repeats_in_utterances[1] = 0
    for i in range(2, num_turns):
        current = filtered[i]  # current utterance
        prev = set.union(*filtered[:i][i%2::i])  # all utterances of the current speaker so far
        repeats = current.intersection(prev)
        num_repeats_in_utterances[i] = len(repeats)

    return num_repeats_in_utterances


def utterance_repetition(conversation):  # called 'word_similarity' in original paper
    """Counts the number of distinct words in the current utterance that were also in the previous utterance"""
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    filtered = [set(utterance).difference(filters) for utterance in
                conversation]  # filter stopwords, contractions and punctuation

    num_repeats_in_utterances[0] = 0  # first utterance can't repeat any word of previous utterance
    for i in range(1, num_turns):
        current = filtered[i]
        prev = filtered[i-1]
        repeats = current.intersection(prev)
        num_repeats_in_utterances[i] = len(repeats)

    return num_repeats_in_utterances


def word_repetition(conversation):  # called 'utterance_repetition' in original paper
    """Counts the number of words that occur multiple times within the same utterance (duplicates) """
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    filtered = [[token for token in utterance if token not in filters] for utterance in conversation]  # filter stopwords, contractions and punctuation

    for i in range(num_turns):
        repeats = len(filtered[i]) - len(set(filtered[i]))  # (difference is positive if a word occurs multiple times)
        num_repeats_in_utterances[i] = repeats

    return num_repeats_in_utterances


def utterance_length(conversation):
    """Counts the length of each utterance."""
    filtered = [[token for token in utterance if token not in punct] for utterance in conversation]  # filter punctuation

    return np.array([len(filtered_utterance) for filtered_utterance in filtered])


# caveats: if the sentiment is negative, it may only be because of the topic, not the person being unhappy with the bot
def deepmoji(conversation):
    """Computes the Deepmoji sentiment of each utterance and
    its (sentiment) coherence with the previous utterance in Deepmoji embedding space"""
    # Init deepmoji just once
    if 'torchmoji' not in globals():
        print('Loading deepmoji')
        from external_metrics_api import DeepmojiAPI
        with torch.no_grad():
            global torchmoji
            torchmoji = DeepmojiAPI()

    # botmoji takes list of utterance strings (not list of lists of tokens)
    utterances = [' '.join(tokens) for tokens in conversation]

    # Run deepmoji: embed utterances
    sentiment_multiplier = _get_sentiment_multiplier()
    utterance_emojis = torchmoji.encode_multiple(utterances)
    sentiments = np.dot(utterance_emojis, sentiment_multiplier)

    # compute coherence (cosine similarity) of each utterance with the previous utterance
    rolled = np.roll(utterance_emojis, shift=1, axis=0)
    emoji_coherence = cosine_similarity(utterance_emojis, rolled)

    # for the first utterance the coherence with the previous utterance cannot be computed --> manually set to zero
    emoji_coherence[0] = 0.0

    return [sentiments, emoji_coherence]


def infersent_coherence(conversation):
    """
    Computes the (semantic) coherence of each utterance with the previous utterance in InferSent embedding space
    (cosine_similarity).
    """
    # Init infersent just once
    if 'infersent' not in globals():
        print('Loading InferSent')
        from external_metrics_api import InferSentAPI
        with torch.no_grad():
            global infersent
            infersent = InferSentAPI()

    utterances = [' '.join(tokens) for tokens in conversation]

    # Run botsent: embed utterances
    embedded_utterances = infersent.encode_multiple(utterances)

    # compute coherence (cosine similarity) of each utterance with the previous utterance
    rolled = np.roll(embedded_utterances, shift=1, axis=0)
    coherence = cosine_similarity(embedded_utterances, rolled)

    # for the first utterance the coherence with the previous utterance cannot be computed --> manually set to zero
    coherence[0] = 0.0
    return coherence


def USE_similarity(conversation):
    """
    Computes the (semantic) coherence of each utterance with the previous utterance in UniversalSentenceEncoder
    embedding space (cosine_similarity).
    Get model:
        1. download model from: https://tfhub.dev/google/universal-sentence-encoder-large/3
        2. unzip at configs.project_dir/UniversalSentenceEncoder
    """
    if 'universal_encoder' not in globals():
        print('Loading Universal Sentence Encoder')
        # import tensorflow as tf
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        import tensorflow_hub as hub
        global universal_encoder, sess, sents, embed_op
        use_path = models_dir.joinpath("UniversalSentenceEncoder/universal-sentence-encoder-large_3")

        with tf.device('/cpu:0'):
            universal_encoder = hub.Module(str(use_path))
            sents = tf.placeholder(tf.string, shape=None, name="input_sents")
            embed_op = universal_encoder(sents)

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    utterances = [' '.join(tokens) for tokens in conversation]

    # Run UniversalSentenceEncoder: embed utterances
    embedded_utterances = sess.run(embed_op, feed_dict={sents: utterances})

    # compute coherence (cosine similarity) of each utterance with the previous utterance
    rolled = np.roll(embedded_utterances, shift=1, axis=0)
    coherence = cosine_similarity(embedded_utterances, rolled)

    # for the first utterance the coherence with the previous utterance cannot be computed --> manually set to zero
    coherence[0] = 0.0
    return coherence


def word2vec_coherence(conversation):
    """
    Computes the coherence of each utterance with the previous utterance in word2vec embedding space (cosine_similarity)
    Get GoogleNews vectors:
        1. download vectors from: https://code.google.com/archive/p/word2vec/
        2. unzip at configs.project_dir/word2vec
    """
    if 'word2vec' not in globals():
        print('Loading word2vec dict')
        import gensim
        global word2vec, keys
        word2vec_path = models_dir.joinpath("word2vec/GoogleNews-vectors-negative300-SLIM.bin")
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        keys = word2vec
        """
        The vocab attribute was removed from KeyedVector in Gensim 4.0.0.
        Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
        """
    num_turns = len(conversation)
    coherence = np.zeros(num_turns)

    # Embed each word in all utterances of the conversation with word2vec and compute the mean embedding per utterance
    # (average over all words in each utterance)
    embedded_utterances = []
    indices_of_utterances_without_embeddings = []
    for idx, utterance in enumerate(conversation):
        embedded_utterance = []
        for word in utterance:  # embed utterance
            if word in keys:
                embedded_utterance.append(word2vec[word])
        if not embedded_utterance:  # no word in the utterance could be embedded (no embedding in word2vec)
            indices_of_utterances_without_embeddings.append(idx)  # save index to set coherence to 0 afterwards
            embedded_utterances.append(np.mean([word2vec['placeholder']], axis=0))  # add placeholder, will be set to 0!
        else:
            embedded_utterances.append(np.mean(embedded_utterance, axis=0))

    embedded_utterances = np.array(embedded_utterances)

    # compute coherence (cosine similarity) of each utterance with the previous utterance
    rolled = np.roll(embedded_utterances, shift=1, axis=0)
    coherence = cosine_similarity(embedded_utterances, rolled)

    # set coherence to 0 for all utterances that didn't get any embedding
    coherence[indices_of_utterances_without_embeddings] = 0.0

    # for the first utterance the coherence with the previous utterance cannot be computed --> manually set to zero
    coherence[0] = 0.0
    return coherence


def empathy(conversation):
    """
    Computes the levels of the three empathy mechanisms (emotional reactions, interpretations, explorations) of
    each utterance with respect to its previous utterance.
    """
    # Init empathy_classifier just once
    if 'empathy_classifier' not in globals():
        # load the empathy classifier
        print('\nLoading empathy_classifier')
        from external_metrics_api import EmpathyClassifierAPI
        with torch.no_grad():
            global empathy_classifier
            empathy_classifier = EmpathyClassifierAPI(device=device)

    utterances = np.array([' '.join(tokens) for tokens in conversation])

    utterance_pairs = np.vstack((np.roll(utterances, shift=1, axis=0), utterances)).T
    # for the first utterance the empathy with the previous utterance cannot be computed
    # --> delete the first pair and add zero empathy level for it later (for all three empathy communication mechanisms)
    utterance_pairs = np.delete(utterance_pairs, 0, 0)

    # Run empathy_classifier: get empathy levels for each utterance pair for each mechanism
    empathy_levels = empathy_classifier.compute_empathy_levels(utterance_pairs)  # shape: (utterance_pair, mechanism)

    # set empathy levels for the first utterance to zero (for all three empathy communication mechanisms)
    empathy_levels = np.vstack((np.zeros(3).astype(int), empathy_levels))

    return empathy_levels  # shape: (utterance, empathy levels)
