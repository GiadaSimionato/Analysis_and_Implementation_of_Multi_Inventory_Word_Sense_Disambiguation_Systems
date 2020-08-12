# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Script that contains the methods to handle the data as to provide the correct input format to the networks.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It creates the vocabularies from the lemmas and from the ids of the senses.
# It shapes the labels and indeces them according to the dictionaries and prepares the inputs with the correct Elmo embedding format.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import random
import numpy as np


# --- Builds the vocabulary from the input sentences (list of lists of lemmas). ---
# :param sentences: list of lists of lemmas (from the XML file)
# :param threshold: a threshold for the relative frequences under which the words are neglected
# :return final_v: final vocabulary 

def get_vocabulary(sentences, threshold):

    tot = 0
    d = dict()
    final_v = ['<PAD>', '<UNK>']
    for sentence in sentences:
        for elem in sentence:
            if elem in d:
                d[elem] += 1
            else:
                d[elem] = 1
            tot += 1
    for key, value in d.items():
        if value/tot>threshold:
            final_v.append(key)
    return final_v


# --- Builds the vocabulary from the senses (list of lists of ids). ---
# :param Ids: list of lists of ids (from the XML file)
# :param drop: a float to describe the percentage of ids to neglet in building the vocabulary (the senses have more or less the same frequency (order 10^-4) so only random choices are possible)
# :return final_vs: final vocabulary 

def get_vocab_senses(Ids, drop):

    vs = list(set(Ids))
    final_vs = []
    for elem in vs:
        if random.randint(1,101)>drop*100:
            final_vs.append(elem)
    return final_vs


# --- Shapes the input as numpy array of length max_len and indices the elements according to the vocabulary. ---
# :param mat: list of lists of input to shape
# :param voc: vocabulary to use for indexing
# :param max_len: maximum length of the sentences
# :return tensor: numpy array of number of rows are the number of sentences gathered from the XML file and number of columns max_len. Each element is a token of the sentences represented as an index (number of word in the vocab)

def indexing(mat, voc, max_len):

    tensor = np.zeros((len(mat), max_len), dtype=int)
    for r, line in enumerate(mat):
        for c, elem in enumerate(line):
            if c<max_len:
                try:
                    tensor[r][c] = voc.index(elem)
                except:
                    tensor[r][c] = 1    # index of <UNK>
    return tensor


# --- Shapes the input as to match the correct format for Elmo embedding. ---
# :param sentences: list of lists of input to shape
# :param max_len: maximum length of the sentences
# :return inputs: tensor of number of rows as the n. of sent. from XML and columns max_len. The elements are tokens of the inputs (eventually padded)
# :return lengths: list of actual lengths of the sentences (for excluding the padding)

def pre_ELMO(sentences, max_len):

    inputs = np.zeros((len(sentences), max_len), dtype=np.dtype('U100'))
    lengths = np.zeros(len(sentences), dtype='int32')
    for row, sentence in enumerate(sentences):
        for column, elem in enumerate(sentence):
            if column<max_len:
                inputs[row][column] = elem
                lengths[row] += 1
    return inputs, lengths
