# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Main script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains an example of the structure that parses the .xml file, shapes the data, creates and trains a model. 
# It's only a TRACE, actual process strongly varied from case to case and it was therefore optimized. 
# The programming part of this work used Google Colab Services.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import nltk
from xml_parser import parse
from utils_mappings import get_dictionary, get_bn2wnDomains
from utils_input import get_vocabulary, get_vocab_senses, indexing, pre_ELMO
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
from tensorflow import keras
from keras.layers import Lambda, Concatenate, Flatten, Multiply, RepeatVector, Permute, Dense, Add, Bidirectional, LSTM, Activation, Dropout, Embedding, Input, CuDNNLSTM, TimeDistributed
from keras.models import load_model, Model
from keras.optimizers import Adam, Adadelta, Nadam
from keras.callbacks import ModelCheckpoint, TensorBoard
from networks import MultitaskBiLSTM, BILSTMElmoSC

threshold_voc = 0.0 
drop_senses = 0 
MAX_LENGTH = 50
EMB_SIZE = 256
HIDDEN_SIZE = 128
DROPOUT = 0.3
DROPOUT_REC = 0.3
BATCH_SIZE = 32
EPOCHS = 10

path_xml = '../SemCor/semcor.data.xml'                  # path of resources needed to build the mappings among various sense representations.
path_inst2sk = '../SemCor/semcor.gold.key.txt'
path_bn2wn = '../resources/babelnet2wordnet.tsv'
path_bn2lex = '../resources/babelnet2lexnames.tsv'
path_bn2wnd = '../resources/babelnet2wndomains.tsv'

inst2sk_dict = get_dictionary(path_inst2sk, 0)          # collection of the dictionaries to realise those mappings
wn2bn_dict = get_dictionary(path_bn2wn, 1)
bn2lex_dict = get_dictionary(path_bn2lex, 0)
bn2wnDom_dict = get_bn2wnDomains(path_bn2wnd)


inputs, labels_BN, bnIds = parse(path_xml, inst2sk_dict, wn2bn_dict, bn2lex_dict, 'BN')     # parsing of the .xml file for retrieving fine-grained (BN) data
_, labels_WND, wndIds = parse(path_xml, inst2sk_dict, wn2bn_dict, bn2wnDom_dict, 'WND')     # parsing of the .xml file for retrieving coarse-grained (WND) data
_, labels_LEX, lexIds = parse(path_xml, inst2sk_dict, wn2bn_dict, bn2lex_dict, 'LN')        # parsing of the .xml file for retrieving coarse-grained (LEX) data

voc_words = get_vocabulary(inputs, 0.0)                         # collect vocabulary of input lemmas

voc_senses_BN = get_vocab_senses(bnIds, 0.0)                    # collect vocabulary of FG senses
voc_joint_BN = voc_words + voc_senses_BN                        # create joint voc.

voc_senses_WND = get_vocab_senses(wndIds, 0.0)                  # collect vocabulary of CG-WND senses
voc_joint_WND = voc_words + voc_senses_WND                      # create joint voc.

voc_senses_LEX = get_vocab_senses(lexIds, 0.0)                  # collect vocabulary of CG-LEX senses
voc_joint_LEX = voc_words + voc_senses_LEX                      # create joint voc.

inputs, max_lengths = pre_ELMO(inputs, MAX_LENGTH)              # prepare the input for matching ELMo embedding requirements

labels_BN = indexing(labels_BN, voc_joint_BN, MAX_LENGTH)       # prepare the BN labels for matching network requirements
labels_BN = np.expand_dims(np.copy(labels_BN), axis=-1)

labels_WND = indexing(labels_WND, voc_joint_WND, MAX_LENGTH)    # prepare the WND labels for matching network requirements
labels_WND = np.expand_dims(np.copy(labels_WND), axis=-1)

labels_LEX = indexing(labels_LEX, voc_joint_LEX, MAX_LENGTH)    # prepare the LEX labels for matching network requirements
labels_LEX = np.expand_dims(np.copy(labels_LEX), axis=-1)

sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)      # needed for downloading ELMo embeddings
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

model_BN = MultitaskBiLSTM((MAX_LENGTH,), len(voc_joint_BN), len(voc_joint_WND), len(voc_joint_LEX), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)     # initialize FG network model
model_WND = BILSTMElmoSC((MAX_LENGTH,), len(voc_joint_WND), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)                                              # initialize CG-WND network model
model_LEX = MultitaskBiLSTM((MAX_LENGTH,), len(voc_joint_BN), len(voc_joint_WND), len(voc_joint_LEX), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)    # initialize CG_LEX network model

# Prepare the callbacks and fit the model with the shaped data

cbk = ModelCheckpoint('mod_BN_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)
logs = TensorBoard(log_dir='../models_logs', batch_size=BATCH_SIZE)
history_BN = model_BN.fit(inputs, labels_BN, batch_size=BATCH_SIZE, callbacks=[cbk, logs], verbose=True, epochs=EPOCHS)

cbk = ModelCheckpoint('mod_WND_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)
logs = TensorBoard(log_dir='../models_logs', batch_size=BATCH_SIZE)
history_WND = model_WND.fit(inputs, labels_BN, batch_size=BATCH_SIZE, callbacks=[cbk, logs], verbose=True, epochs=EPOCHS)

cbk = ModelCheckpoint('mod_LEX_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)
logs = TensorBoard(log_dir='../models_logs', batch_size=BATCH_SIZE)
history_LEX= model_LEX.fit(inputs, labels_BN, batch_size=BATCH_SIZE, callbacks=[cbk, logs], verbose=True, epochs=EPOCHS)
