# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Script that contains the implementations of all the networks
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It defines the structures of all the models used in this work.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import keras.backend as K
from keras.layers import Dense, Bidirectional, LSTM, Activation, Dropout, Embedding, Input, CuDNNLSTM, TimeDistributed, Multiply, Lambda, Add, Flatten, Concatenate, Permute, RepeatVector
from keras.models import load_model, Model
from keras.optimizers import Adam, RMSprop, Adadelta, Nadam

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)      # loads Elmo module
BATCH_SIZE = 32
MAX_LENGTH = 50

# --- Implements the Elmo embedding layer to substitute the vanilla keras one. ---
# :param x: input
# :return None

def ElmoEmbedding(x):

    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(BATCH_SIZE*[MAX_LENGTH])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]

# --- Custom accuracy for the model. ---
# :param y_true: ground truth for the labels
# :param y_pred: predicted version of the labels
# :return sparse categorical accuracy value

def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx())


# --- Baseline: simple stacked Bi-LSTM network. ---
# :param in_shape: shape of the input tensor
# :param vocab_size: size of the vocabulary for chosen task
# :param emb_len: embedding size
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def BILSTM(in_shape, vocab_size, emb_len, hidden_units, dropout, rec_dropout):      # MOD01

    i = Input(shape=in_shape)       # input layer
    embedding = Embedding(input_dim=vocab_size, output_dim=emb_len, input_length=in_shape, mask_zero=True)(i)       # embedding layer
    x = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)    #first bidirectional layer
    x = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(x)        # second bidirectional layer
    out = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(x)     # output layer
    model = Model(input=i, output=out)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Baseline: version of the stacked Bi-LSTM network optimized for training. ---

# :param in_shape: shape of the input tensor
# :param vocab_size: size of the vocabulary for chosen task
# :param emb_len: embedding size
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def CuDNNLSTM(in_shape, vocab_size, emb_len, hidden_units, dropout, rec_dropout):

    i = Input(shape=in_shape)
    embedding = Embedding(input_dim=vocab_size, output_dim=emb_len, input_length=in_shape, mask_zero=True)(i)
    x = Bidirectional(CuDNNLSTM(hidden_units, return_sequences=True), merge_mode='concat')(embedding)
    x = Activation('tanh')(x)
    x = Dropout(rate=dropout)(x)
    x = Bidirectional(CuDNNLSTM(hidden_units, return_sequences=True), merge_mode='concat')(x)
    x = Activation('tanh')(x)
    x = Dropout(rate=dropout)(x)
    out = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(x)
    model = Model(input=i, output=out)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Stacked Bi-LSTM version with custom Elmo embedding (modified version of the first model of the paper). ---

# :param in_shape: shape of the input tensor
# :param vocab_size: size of the vocabulary for chosen task
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model
  
def BILSTMElmo(in_shape, vocab_size, hidden_units, dropout, rec_dropout):        # MOD02
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(i)     # custom Elmo embedding layer
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    out = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(b2)
    model = Model(inputs=i, outputs=out)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Stacked Bi-LSTM version with custom Elmo embedding and a skip connection to boost the training time. ---

# :param in_shape: shape of the input tensor
# :param vocab_size: size of the vocabulary for chosen task
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def BILSTMElmoSC(in_shape, vocab_size, hidden_units, dropout, rec_dropout):      # MOD05
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i)
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])     # skip connection
    out = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(x)
    model = Model(inputs=i, outputs=out)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Stacked Bi-LSTM version with custom Elmo embedding, skip connection and Attention layer (modified version of the second model of the paper). ---

# :param in_shape: shape of the input tensor
# :param vocab_size: size of the vocabulary for chosen task
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def AttentionBiLSTM(in_shape, vocab_size, hidden_units, dropout, rec_dropout):   #MOD06
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i)
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2_out = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2_out])                                
    M = Activation('tanh')(b2_out)       # computes tanh of LSTM hidden state
    u = TimeDistributed(Dense(1))(M)     # weigth vector w implemented as Dense layer with 1 output (eq. 1 first line of the paper)
    u = Flatten()(u)                     # removes additional dimension equal to 1
    a = Activation('softmax')(u)         # normalized attention vector (eq. 1 second line of the paper)
    a = RepeatVector(hidden_units*2)(a)  # to compute eq. 1 third line of the paper
    a = Permute([2,1])(a)                # makes a "transpose" version 
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)     # repeats c over all timesteps to match fig. 2 of the paper
    y = Concatenate()([x, c])           # concatenates c with LSTM output
    out = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(y)
    model = Model(inputs=i, outputs=out)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Stacked Bi-LSTM version with custom Elmo embedding, skip connection and Attention layer in a encoder/decoder structure (modified version of the third model of the paper). ---

# :param in_shape: shape of the input tensor
# :param vocab_size: size of the vocabulary for chosen task
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Seq2Seq(in_shape, vocab_size, hidden_units, dropout, rec_dropout):     #MOD07
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i)
    # Attention Encoder Structure
    b1_enc = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2_enc = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1_enc)
    x = Add()([b1_enc, b2_enc])
    M = Activation('tanh')(b2_enc)
    u = TimeDistributed(Dense(1))(M)
    u = Flatten()(u)
    a = Activation('softmax')(u)
    a = RepeatVector(hidden_units*2)(a)
    a = Permute([2,1])(a)
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)
    # Decoder Structure
    b1_dec = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(c)
    b2_dec = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1_dec)
    out = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(b2_dec)
    model = Model(inputs=i, outputs=out)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()  
    return model


# --- Multitask model for predicting fine-grained, POS and lexicographic name of the instances simultaneously with Attention layer (modified multitask approach of the paper). ---

# :param in_shape: shape of the input tensor
# :param vocab_size_fine: size of the vocabulary for fine-grained task
# :param vocab_size_pos: size of the vocabulary for POS
# :param vocab_size_lexnames: size of the vocabulary for lexnames
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def MultitaskAttentionBiLSTM(in_shape, vocab_size_fine, vocab_size_pos, vocab_size_lexnames, hidden_units, dropout, rec_dropout):      #MOD 081
  
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i)
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    M = Activation('tanh')(b2)  
    u = TimeDistributed(Dense(1))(M)
    u = Flatten()(u)
    a = Activation('softmax')(u)
    a = RepeatVector(hidden_units*2)(a)
    a = Permute([2, 1])(a)
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)
    y = Concatenate()([x, c])
    out_fine_grained = TimeDistributed(Dense(units=vocab_size_fine, activation='softmax'), name='fine_grained')(y)  # output layer for fine grained application
    out_pos = TimeDistributed(Dense(units=vocab_size_pos, activation='softmax'), name='pos')(y)                     # output layer for pos predictions
    out_lexnames = TimeDistributed(Dense(units=vocab_size_lexnames, activation='softmax'), name='lexnames')(y)      # output layer for lexnames application
    model = Model(inputs=i, outputs=[out_fine_grained, out_pos, out_lexnames])
    optimizer = Nadam()
    losses = {'fine_grained': 'sparse_categorical_crossentropy',    # list of losses to minimize
              'pos': 'sparse_categorical_crossentropy',
              'lexnames': 'sparse_categorical_crossentropy'}
    
    lossWeights = {'fine_grained': 1,   # list of weights for the losses
                   'pos': 1,
                   'lexnames': 1}  
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)
    model.summary()
    return model


# --- Model, with Attention layer, that takes as input POS, lexnames and inputs and provides babelnet ids. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_fine: size of the vocabulary for fine-grained task
# :param vocab_size_pos: size of the vocabulary of pos
# :param vocab_size_lex: size of the vocabulary for lexnames
# :param emb_len: embedding size for pos
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Model3(in_shape, vocab_size_fine, vocab_size_pos, vocab_size_lex, emb_len, hidden_units, dropout, rec_dropout):   # part MOD093
  
    i_FG = Input(shape=in_shape[0], dtype=tf.string, name='input_FG')
    i_POS = Input(shape=in_shape[1], name='input_POS')
    i_LEX = Input(shape=in_shape[2], name='input_LEX')
    embedding_FG = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i_FG) 
    embedding_POS = Embedding(input_dim=vocab_size_pos, output_dim=emb_len)(i_POS)
    embedding_LEX = Embedding(input_dim=vocab_size_lex, output_dim=512)(i_LEX)
    concat = Concatenate()([embedding_FG, embedding_POS, embedding_LEX])
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(concat)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    M = Activation('tanh')(b2)  
    u = TimeDistributed(Dense(1))(M)
    u = Flatten()(u)
    a = Activation('softmax')(u)
    a = RepeatVector(hidden_units*2)(a)
    a = Permute([2, 1])(a)
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)
    y = Concatenate()([x, c])  
    out_fine_grained = TimeDistributed(Dense(units=vocab_size_fine, activation='softmax'), name='fine_grained')(y)  
    model = Model(inputs=[i_FG, i_POS, i_LEX], outputs=out_fine_grained)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Multitask model for predicting fine-grained, wordnet domain and lexicographic name of the instances simultaneously. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_fine: size of the vocabulary for fine-grained task
# :param vocab_size_coarse_wndomain: size of the vocabulary for wordnet domains
# :param vocab_size_coarse_lexnames: size of the vocabulary for lexnames
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def MultitaskBiLSTM(in_shape, vocab_size_fine, vocab_size_coarse_wndomain, vocab_size_coarse_lexnames, hidden_units, dropout, rec_dropout):  # MOD 082
  
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(None , 1024))(i)
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    out_fine_grained = TimeDistributed(Dense(units=vocab_size_fine, activation='softmax'), name='fine_grained')(x)                      # output layer for fine-grained
    out_coarse_wndomain = TimeDistributed(Dense(units=vocab_size_coarse_wndomain, activation='softmax'), name='coarse_wndomain')(x)     # output layer for wn domains
    out_coarse_lexnames = TimeDistributed(Dense(units=vocab_size_coarse_lexnames, activation='softmax'), name='coarse_lexnames')(x)     # output layer for lexnames
    model = Model(inputs=i, outputs=[out_fine_grained, out_coarse_wndomain, out_coarse_lexnames])
    optimizer = Nadam()
    losses = {'fine_grained': 'sparse_categorical_crossentropy', 
              'coarse_wndomain': 'sparse_categorical_crossentropy',
              'coarse_lexnames': 'sparse_categorical_crossentropy'}
    lossWeights = {'fine_grained': 1, 
                   'coarse_wndomain': 1,
                   'coarse_lexnames': 1}  
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)
    model.summary() 
    return model


# --- Model, with Attention layer, that takes as input lexnames and inputs and provides babelnet ids. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_fine: size of the vocabulary for fine-grained task
# :param vocab_size_lex: size of the vocabulary for lexnames
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Model2(in_shape, vocab_size_fine, vocab_size_lex, hidden_units, dropout, rec_dropout): # MOD 092
             
    i_INPUT = Input(shape=in_shape[0], dtype=tf.string, name='input_INPUT')
    i_LEX = Input(shape=in_shape[1], name='input_LEX')
    embedding_IN = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i_INPUT)
    embedding_LEX = Embedding(input_dim=vocab_size_lex, output_dim=512)(i_LEX)
    concat = Concatenate()([embedding_IN, embedding_LEX])
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(concat)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    M = Activation('tanh')(b2)
    u = TimeDistributed(Dense(1))(M)
    u = Flatten()(u)
    a = Activation('softmax')(u)
    a = RepeatVector(hidden_units*2)(a)
    a = Permute([2, 1])(a)
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)
    y = Concatenate()([x, c])
    out_fine_grained = TimeDistributed(Dense(units=vocab_size_fine, activation='softmax'), name='fine_grained')(y)
    model = Model(inputs=[i_INPUT, i_LEX], outputs=out_fine_grained)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Version of the multitask model of the paper (in: inputs, out: babelnet, lexnames, pos) without the attention layer. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_fine: size of the vocabulary for fine-grained task
# :param vocab_size_pos: size of the vocabulary for POS
# :param vocab_size_lexnames: size of the vocabulary for lexnames
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Model081NoAtt(in_shape, vocab_size_fine, vocab_size_pos, vocab_size_lexnames, hidden_units, dropout, rec_dropout):   # MOD 081 NO ATTENTION
  
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i)
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    out_fine_grained = TimeDistributed(Dense(units=vocab_size_fine, activation='softmax'), name='fine_grained')(x)
    out_pos = TimeDistributed(Dense(units=vocab_size_pos, activation='softmax'), name='pos')(x)
    out_lexnames = TimeDistributed(Dense(units=vocab_size_lexnames, activation='softmax'), name='lexnames')(x)
    model = Model(inputs=i, outputs=[out_fine_grained, out_pos, out_lexnames])
    optimizer = Nadam()
    losses = {'fine_grained': 'sparse_categorical_crossentropy', 
              'pos': 'sparse_categorical_crossentropy',
              'lexnames': 'sparse_categorical_crossentropy'}
    lossWeights = {'fine_grained': 1, 
                   'pos': 1,
                   'lexnames': 1}  
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)
    model.summary()
    return model


# --- Model, with Attention layer, that takes as input lexnames and inputs and provides wordnet domains. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_wn: size of the vocabulary for wordnet domains
# :param vocab_size_lex: size of the vocabulary for lexnames
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Model095A(in_shape, vocab_size_lex, vocab_size_wn, hidden_units, dropout, rec_dropout):    # MOD 095A

    i_INPUT = Input(shape=in_shape[0], dtype=tf.string, name='input_INPUT')
    i_LEX = Input(shape=in_shape[1], name='input_LEX')        
    embedding_IN = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i_INPUT)
    embedding_LEX = Embedding(input_dim=vocab_size_lex, output_dim=512)(i_LEX)      
    concat = Concatenate()([embedding_IN, embedding_LEX])  
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(concat)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    M = Activation('tanh')(b2)
    u = TimeDistributed(Dense(1))(M)
    u = Flatten()(u)
    a = Activation('softmax')(u)
    a = RepeatVector(hidden_units*2)(a)
    a = Permute([2, 1])(a)
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)
    y = Concatenate()([x, c])
    out_wn = TimeDistributed(Dense(units=vocab_size_wn, activation='softmax'), name='wn_coarse_grained')(y)
    model = Model(inputs=[i_INPUT, i_LEX], outputs=out_wn)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Model, with Attention layer, that takes as input wordnet domains and inputs and provides babelnet predictions. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_wn: size of the vocabulary for wordnet domains
# :param vocab_size_fg: size of the vocabulary for fine-grained task
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Model095B(in_shape, vocab_size_wn, vocab_size_fg, hidden_units, dropout, rec_dropout):     #MOD095B

    i_INPUT = Input(shape=in_shape[0], dtype=tf.string, name='input_INPUT')
    i_WND = Input(shape=in_shape[1], name='input_WND')          
    embedding_IN = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i_INPUT)
    embedding_WND = Embedding(input_dim=vocab_size_wn, output_dim=512)(i_WND)    
    concat = Concatenate()([embedding_IN, embedding_WND])
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(concat)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    M = Activation('tanh')(b2)
    u = TimeDistributed(Dense(1))(M)
    u = Flatten()(u)
    a = Activation('softmax')(u)
    a = RepeatVector(hidden_units*2)(a)
    a = Permute([2, 1])(a)
    c = Multiply()([x, a])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(2*hidden_units,))(c)
    c = RepeatVector(MAX_LENGTH)(c)
    y = Concatenate()([x, c])
    out_fine_grained = TimeDistributed(Dense(units=vocab_size_fg, activation='softmax'), name='fine_grained')(y)
    model = Model(inputs=[i_INPUT, i_WND], outputs=out_fine_grained)
    optimizer = Nadam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[custom_sparse_categorical_accuracy])
    model.summary()
    return model


# --- Model that provides POS and lexnames predictions. ---

# :param in_shape: shape of the input tensor
# :param vocab_size_pos: size of the vocabulary for POS
# :param vocab_size_lexnames: size of the vocabulary for lexnames
# :param hidden_units: number of hidden units for the bidirectional layers
# :param dropout: dropout rate
# :param rec_dropout: recurrent dropout rate
# :return model: the model

def Model4(in_shape, vocab_size_pos, vocab_size_lexnames, hidden_units, dropout, rec_dropout):     # MOD 094
  
    i = Input(shape=in_shape, dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LENGTH, 1024))(i)
    b1 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(embedding)
    b2 = Bidirectional(LSTM(hidden_units, dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True), merge_mode='concat')(b1)
    x = Add()([b1, b2])
    out_pos = TimeDistributed(Dense(units=vocab_size_pos, activation='softmax'), name='pos')(x)
    out_lexnames = TimeDistributed(Dense(units=vocab_size_lexnames, activation='softmax'), name='lexnames')(x)
    model = Model(inputs=i, outputs=[out_pos, out_lexnames])
    optimizer = Nadam()
    losses = {'pos': 'sparse_categorical_crossentropy',
              'lexnames': 'sparse_categorical_crossentropy'}
    lossWeights = {'pos': 1,
                   'lexnames': 1}  
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)
    model.summary()
    return model