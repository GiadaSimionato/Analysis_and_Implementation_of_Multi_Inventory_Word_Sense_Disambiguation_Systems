# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Script that contains additional versions of the predict function implemented to try different configuration in the boosting phases
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# Thess methods are not commented due to their additional not required nature.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------




from xml_parser import blind_parsing
import pickle
import keras
from keras.models import load_model, Model
import keras.backend as K
from utils_input import indexing, pre_ELMO
from utils_mappings import get_dictionary, get_synsetsIds, get_MFS, convert_bn2wnd, convert_bn2lex
import numpy as np
from networks import BILSTMElmo, BILSTM, BILSTMElmoSC, Seq2Seq, AttentionBiLSTM, MultitaskAttentionBiLSTM, MultitaskBiLSTM, Model3, Model2, Model095A, Model095B, Model4
from predict import crop, ext, get_best, convert_bn2lex, restrict_index_synsets
from keras.preprocessing.sequence import pad_sequences

threshold_voc = 0.0 
drop_senses = 0 
MAX_LENGTH = 50  
EMB_SIZE = 16
HIDDEN_SIZE = 128
DROPOUT = 0.3
DROPOUT_REC = 0.3
BATCH_SIZE = 32
EPOCHS = 4

def get_best_pos(preds, voc):

    best_index = np.argmax(preds)
    return voc[best_index+2]


def filter_with_LEX(set_syn, bestLex, bn2lex_dict):

    new_set = []
    set_syn = list(set(set_syn))
    set_lex = convert_bn2lex(set_syn, bn2lex_dict)
    for i, elem in enumerate(set_lex):
        if elem == bestLex:
            new_set.append(set_syn[i])
    return new_set

def filter_with_WND(set_syn, bestWnd, bn2wnd_dict):

    new_set = []
    set_syn = list(set(set_syn))
    set_wnd = convert_bn2wnd(set_syn, bn2wnd_dict)
    for i, elem in enumerate(set_wnd):
        if elem == bestWnd:
            new_set.append(set_syn[i])
    return new_set

def predict_babelnet_mod091(input_path : str, output_path : str, resources_path : str) -> None:

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/vocabulary', 'rb') as f:
        voc_BN = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)
    
    model_in2lex = AttentionBiLSTM((max_len,), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2lex.load_weights(resources_path+'/mod06_LEX_weights.h5')
    model_in2bn = Seq2Seq((max_len,), len(voc_BN), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2bn.load_weights(resources_path+'/mod07_BN_weights.h5')

    inputs = crop(inputs, max_len)
    pos = crop(pos, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    pos = ext(pos, max_len, batch_size, 'pos')
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    inputs, _ = pre_ELMO(inputs, max_len)

    pred_LEX = model_in2lex.predict(inputs, batch_size=batch_size, verbose=True)
    pred_BN = model_in2bn.predict(inputs, batch_size=batch_size, verbose=True)

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)

                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(pred_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                
                pox_synsets = filter_with_LEX(pox_synsets, bestLEX, bn2lex_dict)    # pox synsets restricted with lex
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)

                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(pred_BN[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    
    pass


def predict_babelnet_mod092(input_path : str, output_path : str, resources_path : str) -> None:

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/vocabulary', 'rb') as f:
        voc_BN = pickle.load(f)

    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)

    model_in2lex = AttentionBiLSTM((max_len,), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2lex.load_weights(resources_path+'/mod06_LEX_weights.h5')

    inputs = crop(inputs, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    pos = crop(pos, max_len)
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    pos = ext(pos, max_len, batch_size, 'pos')
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    flags = pad_sequences(flags, maxlen=max_len, dtype='bool', padding='post', value=False)
    inputs, _ = pre_ELMO(inputs, max_len)

    preds_LEX = model_in2lex.predict(inputs, batch_size=batch_size, verbose=True)

    inputs_LEX = []
    
    for r, sentence in enumerate(flags):    # for each sentence
        inputs_LEX_part = []
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict) 

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)
                
                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(preds_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                    
                bestLEX = voc_LEX.index(bestLEX)
            else:

                bestLEX = np.argmax(preds_LEX[r][c])
            inputs_LEX_part.append(bestLEX)
        inputs_LEX.append(inputs_LEX_part)

    inputs_LEX = np.asarray(inputs_LEX)
    
    model_inLex2bn = Model2([(max_len,), inputs_LEX.shape[1:2]], len(voc_BN), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_inLex2bn.load_weights(resources_path+'/mod092_weights.h5')
    
    predictions = model_inLex2bn.predict([inputs, inputs_LEX], batch_size=batch_size, verbose=True)
    
    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated
                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)
                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(predictions[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    
    pass


def predict_babelnet_mod093(input_path : str, output_path : str, resources_path : str) -> None:

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, _, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/vocabulary', 'rb') as f:
        voc_BN = pickle.load(f)
    with open(resources_path +'/vocabularyPOS', 'rb') as f:
        voc_POS = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)

    model_in2lex = AttentionBiLSTM((max_len,), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2lex.load_weights(resources_path+'/mod06_LEX_weights.h5')
    model_in2pos = BILSTMElmoSC((max_len,), len(voc_POS), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2pos.load_weights(resources_path+'/mod_POS_weights.h5')

    inputs = crop(inputs, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    flags = pad_sequences(flags, maxlen=max_len, dtype='bool', padding='post', value=False)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    inputs, _ = pre_ELMO(inputs, max_len)

    preds_LEX = model_in2lex.predict(inputs, batch_size=batch_size, verbose=True)
    preds_POS = model_in2pos.predict(inputs, batch_size=batch_size, verbose=True)

    inputs_POS = []
    inputs_LEX = []
    
    for r, sentence in enumerate(flags):    # for each sentence
        inputs_POS_part = []
        inputs_LEX_part = []
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                bestPOS = get_best_pos(preds_POS[r][c][2:], voc_POS)    # get best pos
                bestPOS = voc_POS.index(bestPOS)
                pox_synsets = get_synsetsIds(inputs[r][c], bestPOS, wn2bn_dict) 

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)
                
                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(preds_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                bestLEX = voc_LEX.index(bestLEX)
            else:

                bestPOS = np.argmax(preds_POS[r][c])
                bestLEX = np.argmax(preds_LEX[r][c])

            inputs_POS_part.append(bestPOS)
            inputs_LEX_part.append(bestLEX)
        inputs_POS.append(inputs_POS_part)
        inputs_LEX.append(inputs_LEX_part)

    inputs_POS = np.asarray(inputs_POS)
    inputs_LEX = np.asarray(inputs_LEX)
    #inputs_POS = np.expand_dims(inputs_POS, axis=-1)
    #inputs_LEX = np.expand_dims(inputs_LEX, axis=-1)

    model_inLexPos2bn =  Model3Modified([(max_len,), inputs_POS.shape[1:2], inputs_LEX.shape[1:2]], len(voc_BN), len(voc_POS), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_inLexPos2bn.load_weights(resources_path+'/mod093_modified_weights.h5')
    
    predictions = model_inLexPos2bn.predict([inputs, inputs_POS, inputs_LEX], batch_size=batch_size, verbose=True)

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated
                pox_synsets = get_synsetsIds(inputs[r][c], inputs_POS[r][c], wn2bn_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)
                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(predictions[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    pass

def predict_babelnet_mod094(input_path : str, output_path : str, resources_path : str) -> None:

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, _, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/vocabulary', 'rb') as f:
        voc_BN = pickle.load(f)
    with open(resources_path +'/vocabularyPOS', 'rb') as f:
        voc_POS = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)

    model_in2lexPos = Model4((max_len,), len(voc_POS), len(voc_LEX), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2lexPos.load_weights(resources_path+'/mod094_weights.h5')

    inputs = crop(inputs, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    flags = pad_sequences(flags, maxlen=max_len, dtype='bool', padding='post', value=False)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    inputs, _ = pre_ELMO(inputs, max_len)

    preds_LEX = model_in2lexPos.predict(inputs, batch_size=batch_size, verbose=True)[1]
    preds_POS = model_in2lexPos.predict(inputs, batch_size=batch_size, verbose=True)[0]

    inputs_POS = []
    inputs_LEX = []
    
    for r, sentence in enumerate(flags):    # for each sentence
        inputs_POS_part = []
        inputs_LEX_part = []
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                bestPOS = get_best_pos(preds_POS[r][c][2:], voc_POS)    # get best pos
                bestPOS = voc_POS.index(bestPOS)
                pox_synsets = get_synsetsIds(inputs[r][c], bestPOS, wn2bn_dict) 

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)
                
                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(preds_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                bestLEX = voc_LEX.index(bestLEX)
            else:

                bestPOS = np.argmax(preds_POS[r][c])
                bestLEX = np.argmax(preds_LEX[r][c])

            inputs_POS_part.append(bestPOS)
            inputs_LEX_part.append(bestLEX)
        inputs_POS.append(inputs_POS_part)
        inputs_LEX.append(inputs_LEX_part)

    inputs_POS = np.asarray(inputs_POS)
    inputs_LEX = np.asarray(inputs_LEX)

    model_inLexPos2bn =  Model3Modified([(max_len,), inputs_POS.shape[1:2], inputs_LEX.shape[1:2]], len(voc_BN), len(voc_POS), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_inLexPos2bn.load_weights(resources_path+'/mod093_modified_weights.h5')
    
    predictions = model_inLexPos2bn.predict([inputs, inputs_POS, inputs_LEX], batch_size=batch_size, verbose=True)

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated
                pox_synsets = get_synsetsIds(inputs[r][c], inputs_POS[r][c], wn2bn_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)
                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(predictions[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()

    pass

def predict_babelnet_mod095(input_path : str, output_path : str, resources_path : str) -> None:

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/vocabulary', 'rb') as f:
        voc_BN = pickle.load(f)
    with open(resources_path+'/vocabularyWND', 'rb') as f:
        voc_WND = pickle.load(f)

    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)
    bn2wnd_dict = get_dictionary(resources_path+'/babelnet2wndomains.tsv', 0)

    model_in2lex = AttentionBiLSTM((max_len,), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2lex.load_weights(resources_path+'/mod06_LEX_weights.h5')

    inputs = crop(inputs, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    pos = crop(pos, max_len)
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    pos = ext(pos, max_len, batch_size, 'pos')
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    flags = pad_sequences(flags, maxlen=max_len, dtype='bool', padding='post', value=False)
    inputs, _ = pre_ELMO(inputs, max_len)

    preds_LEX = model_in2lex.predict(inputs, batch_size=batch_size, verbose=True)

    inputs_LEX = []
    
    for r, sentence in enumerate(flags):    # for each sentence
        inputs_LEX_part = []
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict) 

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)
                
                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(preds_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                    
                bestLEX = voc_LEX.index(bestLEX)
            else:

                bestLEX = np.argmax(preds_LEX[r][c])
            inputs_LEX_part.append(bestLEX)
        inputs_LEX.append(inputs_LEX_part)

    inputs_LEX = np.asarray(inputs_LEX)
    
    model_inLex2wn = Model095A([(max_len,), inputs_LEX.shape[1:2]], len(voc_LEX), len(voc_WND), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_inLex2wn.load_weights(resources_path+'/mod095A_weights.h5')
    
    preds_WND = model_inLex2wn.predict([inputs, inputs_LEX], batch_size=batch_size, verbose=True)
    
    inputs_WND = []
    
    for r, sentence in enumerate(flags):    # for each sentence
        inputs_WND_part = []
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict) 

                pox_WND = convert_bn2wnd(pox_synsets, bn2wnd_dict)
                pox_WND = restrict_index_synsets(pox_WND, voc_WND)
                
                if len(pox_WND)==0:
                    bestWND = get_MFS(inputs[r][c], wn2bn_dict)
                    bestWND = convert_bn2lex([bestWND], bn2wnd_dict)[0]
                else:
                    bestWND = get_best(preds_WND[r][c], pox_WND, voc_WND)      # best lex cathegory
                    
                bestWND = voc_WND.index(bestWND)
            else:

                bestWND = np.argmax(preds_WND[r][c])
            inputs_WND_part.append(bestWND)
        inputs_WND.append(inputs_WND_part)

    inputs_WND = np.asarray(inputs_WND)

    model_inWnd2bn = Model095B([(max_len,), inputs_LEX.shape[1:2]], len(voc_WND), len(voc_BN), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_inWnd2bn.load_weights(resources_path+'/mod095B_weights.h5')
    
    predictions = model_inWnd2bn.predict([inputs, inputs_WND], batch_size=batch_size, verbose=True)

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated
                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)
                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(predictions[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    pass

def predict_babelnet_mod096(input_path : str, output_path : str, resources_path : str) -> None:

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/vocabulary', 'rb') as f:
        voc_BN = pickle.load(f)
    with open(resources_path+'/vocabularyWND', 'rb') as f:
        voc_WND = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)
    bn2wnd_dict = get_dictionary(resources_path+'/babelnet2wndomains.tsv', 0)
    
    model_in2lex = AttentionBiLSTM((max_len,), len(voc_LEX), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2lex.load_weights(resources_path+'/mod06_LEX_weights.h5')
    model_in2bn = Seq2Seq((max_len,), len(voc_BN), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2bn.load_weights(resources_path+'/mod07_BN_weights.h5')
    model_in2wnd = BILSTMElmoSC((max_len,), len(voc_WND), EMB_SIZE, HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model_in2wnd.load_weights(resources_path+'/mod05_WND_weights.h5')

    inputs = crop(inputs, max_len)
    pos = crop(pos, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    pos = ext(pos, max_len, batch_size, 'pos')
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    inputs, _ = pre_ELMO(inputs, max_len)

    pred_LEX = model_in2lex.predict(inputs, batch_size=batch_size, verbose=True)
    pred_BN = model_in2bn.predict(inputs, batch_size=batch_size, verbose=True)
    pred_WND = model_in2wnd.predict(inputs, batch_size=batch_size, verbose=True)

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)

                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(pred_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                
                pox_WND = convert_bn2wnd(pox_synsets, bn2wnd_dict)
                pox_WND = restrict_index_synsets(pox_WND, voc_WND)

                if len(pox_WND)==0:
                    bestWND = get_MFS(inputs[r][c], wn2bn_dict)
                    bestWND = convert_bn2wnd([bestWND], bn2wnd_dict)[0]
                else:
                    bestWND = get_best(pred_WND[r][c], pox_WND, voc_WND)      # best lex cathegory


                pox_synsets = filter_with_LEX(pox_synsets, bestLEX, bn2lex_dict)    # pox synsets restricted with lex
                pox_synsets = filter_with_WND(pox_synsets, bestWND, bn2wnd_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)

                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(pred_BN[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    
    pass



def predict_babelnet_mod096_MODIFIED(input_path : str, output_path : str, resources_path : str) -> None:
    
    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml

    with open(resources_path+'/OMSTI_voc_LEX', 'rb') as f:
        voc_LEX = pickle.load(f)
    with open(resources_path+'/OMSTI_voc_BN', 'rb') as f:
        voc_BN = pickle.load(f)
    with open(resources_path+'/OMSTI_voc_WND', 'rb') as f:
        voc_WND = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)
    bn2wnd_dict = get_dictionary(resources_path+'/babelnet2wndomains.tsv', 0)

    model = MultitaskBiLSTM((max_len,), len(voc_BN), len(voc_WND), len(voc_LEX), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model.load_weights(resources_path+'/mod_OMSTI_082_weights.h5')

    inputs = crop(inputs, max_len)
    pos = crop(pos, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    pos = ext(pos, max_len, batch_size, 'pos')
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    inputs, _ = pre_ELMO(inputs, max_len)
  
    pred_BN, pred_WND, pred_LEX = model.predict(inputs, batch_size=batch_size, verbose=True)
    

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)

                if len(pox_LEX)==0:
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0]
                else:
                    bestLEX = get_best(pred_LEX[r][c], pox_LEX, voc_LEX)      # best lex cathegory
                
                pox_WND = convert_bn2wnd(pox_synsets, bn2wnd_dict)
                pox_WND = restrict_index_synsets(pox_WND, voc_WND)

                if len(pox_WND)==0:
                    bestWND = get_MFS(inputs[r][c], wn2bn_dict)
                    bestWND = convert_bn2wnd([bestWND], bn2wnd_dict)[0]
                else:
                    bestWND = get_best(pred_WND[r][c], pox_WND, voc_WND)      # best lex cathegory


                pox_synsets = filter_with_LEX(pox_synsets, bestLEX, bn2lex_dict)    # pox synsets restricted with lex
                pox_synsets = filter_with_WND(pox_synsets, bestWND, bn2wnd_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)

                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                else:
                    best = get_best(pred_BN[r][c], pox_synsets, voc_BN)
                # write best on external sheet
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    
    pass