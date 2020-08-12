from xml_parser import blind_parsing
import pickle
import keras
from keras.models import load_model, Model
import keras.backend as K
from utils_input import indexing, pre_ELMO
from utils_mappings import get_dictionary, get_synsetsIds, get_MFS, convert_bn2wnd, convert_bn2lex
import numpy as np
from networks import BILSTMElmo, BILSTM, BILSTMElmoSC, Seq2Seq, AttentionBiLSTM, MultitaskAttentionBiLSTM, MultitaskBiLSTM, Model081NoAtt


HIDDEN_SIZE = 128
DROPOUT = 0.3
DROPOUT_REC = 0.3

# --- Methods for crop the sentences at the maximum length without loosing data. ---
# :param inputs: input to crop
# :param maxLength: maximum length at which to crop
# :return output: cropped version of the input, eventually padded

def crop(inputs, maxLength):
    output = []
    for line in inputs:
        ratio = int(np.ceil(len(line)/maxLength))
        for times in range(ratio-1):
            output.append(line[times*maxLength:((times+1)*maxLength)])
        portion = line[maxLength*(ratio-1):]
        output.append(portion)
    return output


# --- Methods for extending the input in such a way to match a multiple of the batch-size to handle the Keras bug. ---
# :param inputs: input to extend
# :param maxLength: max length of the input
# :param batch_size: batch size
# :param control: to understand which tipe of neutral element it's needed to extend the input (to match input format)
# :return inputs: the extended version of the input

def ext(inputs, maxLength, batch_size, control):

    n_samples = len(inputs)
    n_toCover = batch_size - n_samples%batch_size
    
    if control == 'flags':
        empty = maxLength*[False]
    else:
        empty = maxLength*['']
    for i in range(n_toCover):
        inputs.append(empty)
    return inputs


# --- Method to select the synset among a list that are in the given vocabulary. ---
# :param set_syn: list of synsets
# :param voc: vocabulary
# :return pox: list of synsets that are in the vocabulary

def restrict_index_synsets(set_syn, voc):
    set_syn = list(set(set_syn))
    pox = []
    for elem in set_syn:
        try:
            index = voc.index(elem)
            pox.append(index)
        except:
            continue
    return pox


# --- Method that filters the candidate synsets to only those belonging to a specific LEX category. ---
# :param set_syn: list of candidate synsets
# :param bestLex: best LEX category
# :param bn2lex_dict: dictionary babelnet ids to lexnames
# :return new_set: list of restricted synsets

def filter_with_LEX(set_syn, bestLex, bn2lex_dict):

    new_set = []
    set_syn = list(set(set_syn))
    set_lex = convert_bn2lex(set_syn, bn2lex_dict)
    for i, elem in enumerate(set_lex):
        if elem == bestLex:
            new_set.append(set_syn[i])
    return new_set


# --- Method that filters the candidate synsets to only those belonging to a specific WND category. ---
# :param set_syn: list of candidate synsets
# :param bestWnd: best WND category
# :param bn2wnd_dict: dictionary babelnet ids to WordNet domains
# :return new_set: list of restricted synsets

def filter_with_WND(set_syn, bestWnd, bn2wnd_dict):

    new_set = []
    set_syn = list(set(set_syn))
    set_wnd = convert_bn2wnd(set_syn, bn2wnd_dict)
    for i, elem in enumerate(set_wnd):
        if elem == bestWnd:
            new_set.append(set_syn[i])
    return new_set


# -- Retrieve best sense for a word based on predictions.
# :param preds: list of probabilities for a specific word
# :param pox_synsets: list of indeces of the synsets into the vocabulary voc
# :param voc: vocabulary
# :return best: best sense of a given word

def get_best(preds, pox_synsets, voc):

    probs = []
    for elem in pox_synsets:
        probs.append(preds[elem])
    best_index = np.argmax(np.asarray(probs))
    best = voc[pox_synsets[best_index]]
    return best

def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx())


def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml (inputs: list of lists of lemmas, pos: list of lists of POS, flags: 
                                                            # list of lists of boolean (True:to disamb, False otherwise) and ids: list of lists of ids)

    with open(resources_path+'/OMSTI_voc_LEX', 'rb') as f:  # retrieving vocabularies
        voc_LEX = pickle.load(f)
    with open(resources_path+'/OMSTI_voc_BN', 'rb') as f:
        voc_BN = pickle.load(f)
    with open(resources_path+'/OMSTI_voc_WND', 'rb') as f:
        voc_WND = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)    # retrieving mappings
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)
    bn2wnd_dict = get_dictionary(resources_path+'/babelnet2wndomains.tsv', 0)

    model = MultitaskBiLSTM((max_len,), len(voc_BN), len(voc_WND), len(voc_LEX), HIDDEN_SIZE, DROPOUT, DROPOUT_REC) # initialize the model
    model.load_weights(resources_path+'/mod_OMSTI_082_weights.h5')    # load weights

    # Shaping the inputs, pos, flags and ids to match ELMo and network requirements
    inputs = crop(inputs, max_len)  
    pos = crop(pos, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    pos = ext(pos, max_len, batch_size, 'pos')
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    inputs, _ = pre_ELMO(inputs, max_len)

    pred_BN, pred_WND, pred_LEX = model.predict(inputs, batch_size=batch_size, verbose=True)    # predict

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:                        # flag = True, that word must be disambiguated

                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)   #retrieve candidate synsets

                pox_LEX = convert_bn2lex(pox_synsets, bn2lex_dict)
                pox_LEX = restrict_index_synsets(pox_LEX, voc_LEX)      # restrict pox synsets to only those belonging to the vocabulary

                if len(pox_LEX)==0:                                     # if non is present
                    bestLEX = get_MFS(inputs[r][c], wn2bn_dict)         # backoff strategy
                    bestLEX = convert_bn2lex([bestLEX], bn2lex_dict)[0] # get LEX category of bakeoff strategy
                else:
                    bestLEX = get_best(pred_LEX[r][c], pox_LEX, voc_LEX)      # best LEX cathegory
                
                pox_WND = convert_bn2wnd(pox_synsets, bn2wnd_dict)      
                pox_WND = restrict_index_synsets(pox_WND, voc_WND)      # restrict pox synsets to only those belonging to the vocabulary

                if len(pox_WND)==0:
                    bestWND = get_MFS(inputs[r][c], wn2bn_dict)         # backoff strategy
                    bestWND = convert_bn2wnd([bestWND], bn2wnd_dict)[0] # get WND category of bakeoff strategy
                else:
                    bestWND = get_best(pred_WND[r][c], pox_WND, voc_WND)      # best WND cathegory


                pox_synsets = filter_with_LEX(pox_synsets, bestLEX, bn2lex_dict)    # pox synsets restricted with LEX
                pox_synsets = filter_with_WND(pox_synsets, bestWND, bn2wnd_dict)    # pox synsets restricted with WND
                pox_synsets = restrict_index_synsets(pox_synsets, voc_BN)

                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)            # backoff strategy
                else:
                    best = get_best(pred_BN[r][c], pox_synsets, voc_BN) # best synset like BabelNet id
                out_file.write(ids[r][c] + ' ' + best + '\n')   # write on the external file matching the format required
    out_file.close()
    
    pass


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml
    
    with open(resources_path+'/vocabulary', 'rb') as f:
        vocFG = pickle.load(f)                              # retrieve vocabulary (joint)
    
    with open(resources_path+'/vocabularyWND', 'rb') as f:
        vocWND = pickle.load(f)
                                                        
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2wnd_dict = get_dictionary(resources_path+'/babelnet2wndomains.tsv', 0)

    model = BILSTMElmoSC((max_len,), len(vocWND), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model.load_weights(resources_path+'/mod05_WND_weights.h5')

    inputs = crop(inputs, max_len)
    pos = crop(pos, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    pos = ext(pos, max_len, batch_size, 'pos')
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    inputs, _ = pre_ELMO(inputs, max_len)

    predictions = model.predict(inputs, batch_size=batch_size, verbose=True)

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:                        # flag = True, that word must be disambiguated
                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)   
                pox_synsets = convert_bn2wnd(pox_synsets, bn2wnd_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, vocWND)
                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                    best = convert_bn2wnd([best], bn2wnd_dict)[0]
                else:
                    best = get_best(predictions[r][c], pox_synsets, vocWND)
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    pass


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    batch_size = 32
    max_len= 50

    out_file = open(output_path, 'w', encoding='utf-8')
    inputs, pos, flags, ids = blind_parsing(input_path)     # parsing xml
    
    with open(resources_path+'/vocabulary', 'rb') as f:
        vocFG = pickle.load(f)
    with open(resources_path+'/vocabularyWND', 'rb') as f:
        vocWND = pickle.load(f)
    with open(resources_path+'/vocabularyLEX', 'rb') as f:
        vocLEX = pickle.load(f)
    
    wn2bn_dict = get_dictionary(resources_path+'/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary(resources_path+'/babelnet2lexnames.tsv', 0)

    model = MultitaskBiLSTM((max_len,), len(vocFG), len(vocWND), len(vocLEX), HIDDEN_SIZE, DROPOUT, DROPOUT_REC)
    model.load_weights(resources_path+'/mod082_weights.h5')

    inputs = crop(inputs, max_len)
    pos = crop(pos, max_len)
    flags = crop(flags, max_len)
    ids = crop(ids, max_len)
    pos = ext(pos, max_len, batch_size, 'pos')
    flags = ext(flags, max_len, batch_size, 'flags')
    ids = ext(ids, max_len, batch_size, 'ids')
    inputs = ext(inputs, max_len, batch_size, 'inputs')
    inputs, _ = pre_ELMO(inputs, max_len)
    
    predictions = model.predict(inputs, batch_size=batch_size, verbose=True)[2]

    for r, sentence in enumerate(flags):    # for each sentence
        for c, word in enumerate(sentence): # for each word
            if word:   # flag = True, that word must be disambiguated
                pox_synsets = get_synsetsIds(inputs[r][c], pos[r][c], wn2bn_dict)
                pox_synsets = convert_bn2lex(pox_synsets,bn2lex_dict)
                pox_synsets = restrict_index_synsets(pox_synsets, vocLEX)
                if len(pox_synsets)==0:
                    best = get_MFS(inputs[r][c], wn2bn_dict)
                    best = convert_bn2lex([best], bn2lex_dict)[0]
                else:
                    best = get_best(predictions[r][c], pox_synsets, vocLEX)
                out_file.write(ids[r][c] + ' ' + best + '\n')
    out_file.close()
    pass
