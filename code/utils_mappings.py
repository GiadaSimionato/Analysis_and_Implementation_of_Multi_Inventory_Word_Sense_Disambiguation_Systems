# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Script that contains the methods to handle the mappings among the various representations of the senses.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It creates the bridges among all the various representation: babelnet ids, wordnet ids, lexnames, wordnet domains, sensekeys and instance ids.
# It also provides a way for retrieving the senses of a lemma, even tied with POS, or its most frequent sense.
# It contains methods for converting the gold files of the evaluation dataset into the format that matches the one required to compute the score.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

from nltk.corpus import wordnet as wn

# --- Creates the dictionary for mapping two representations of the senses from a specific file path. ---
# :param path: the path of the file in tsv format from which to collect the data
# :param a: the direction of the mapping, a=0 for normal and a= 1. Used for: bn2wn; bn2lexnames; instancesId2sensekeys;
# :return d: dictionary between two representation

def get_dictionary(path, a): 

    d = dict()
    f = open(path, encoding='utf-8')
    line = f.readline()
    while line!='':
        parts = line.split()
        d[parts[0+a]] = parts[1-a].strip()
        line = f.readline()
    f.close()
    return d


# --- Creates the dictionary that maps babelnet ids to wordnet Domains. ---
# :param path: path of the file for this mapping
# :return d: dictionary with babelnet ids as keys and wordnet domains as values

def get_bn2wnDomains(path):  

    d = dict()
    f = open(path, encoding='utf-8')
    line = f.readline()
    while line!='':
        parts = line.split('\t')
        value = []
        for i in range(1,len(parts)):
            value.append(parts[i].strip())
        d[parts[0]] = value
        line = f.readline()
    f.close()
    return d


# --- Gets wordnet id from sensekey. ---
# :param sensekey: sensekey
# :return synset_id: the corresponding wordnet id 

def sensekey2wn(sensekey): 

    synset = wn.lemma_from_key(sensekey).synset()
    synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
    return synset_id


# --- Gets list of babelnet ids of all candidate synsets of a specific lemma restricted with POS (whenever possible). ---
# :param lemma: lemma from which to retrieve senses
# :param POS: corresponding POS
# :param wn2bn_dict: dictionary with keys wordnet ids and values the corresponding babelnet ones
# :return ids: list of babelnet ids of the candidate synsets

def get_synsetsIds(lemma, POS, wn2bn_dict):

    try:
        l = wn.synsets(lemma, pos=getattr(wn, POS)) # to match the format of wordnet
    except:
        l = wn.synsets(lemma)                       # in case there is no synset with a certain POS
    ids = []
    for elem in l:
        wnId = "wn:" + str(elem.offset()).zfill( 8) + elem.pos()
        bnId = wn2bn_dict[wnId]
        ids.append(bnId)
    return ids


# --- Gets the babelnet id of the most frequent sense of a lemma. ---
# :param lemma: lemma
# :param wn2bn_dict: dictionary with keys wordnet ids and values the corresponding babelnet ones
# :return bnMFS: babelnet id of the MFS

def get_MFS(lemma, wn2bn_dict):

    mfs = wn.synsets(lemma)[0]
    wnMFS = "wn:" + str(mfs.offset()).zfill( 8) + mfs.pos()
    bnMFS = wn2bn_dict[wnMFS]
    return bnMFS


# --- Gets babelnet ids from the instance ids. ---
# :param instanceId: instance to transform
# :param inst2sk_dict: dictionary whose keys are the instance ids and the values are the sensekeys
# :param wn2bn_dict: dictionary with keys wordnet ids and values the corresponding babelnet ones
# :return bnId: the corresponding babelnet id

def inst2bn(instanceId, inst2sk_dict, wn2bn_dict):

    senseKey = inst2sk_dict[instanceId]
    wnId = sensekey2wn(senseKey)
    bnId = wn2bn_dict[wnId]
    return bnId


# --- Gets the coarse-grained ids from the instance ids. ---
# :param instanceId: instance to transform
# :param inst2sk_dict: dictionary whose keys are the instance ids and the values are the sensekeys
# :param wn2bn_dict: dictionary with keys wordnet ids and values the corresponding babelnet ones
# :param bn2cg_dict: dictionary with keys the babelnet ids and values the coarse-grained ids (wordnet domains or lexnames)
# :param control: if 'WND' it assigns 'factotum' whenever there is no mapping between babelnet and wordnet domains
# :return cgId: coarse-grained id

def inst2cg(instanceId, inst2sk_dict, wn2bn_dict, bn2cg_dict, control):

    senseKey = inst2sk_dict[instanceId]
    wnId = sensekey2wn(senseKey)
    bnId = wn2bn_dict[wnId]
    try:
        cgId = bn2cg_dict[bnId]
    except:
        if control=='WND':
            cgId = 'factotum'
    return cgId


# --- Converts a list of babelnet ids to a list of corresponding wordnet domains. ---
# :param bnIds: list of babelnet ids
# :param bn2wnd_dict: dictionary with keys babelnet ids and values the corresponding wordnet ones
# :return wndIds: corresponding list of wordnet ids

def convert_bn2wnd(bnIds, bn2wnd_dict):

    wndIds = []
    for elem in bnIds:
        try:
            wndId = bn2wnd_dict[elem]
        except:                         # if the mapping is not present
            wndId = 'factotum'
        wndIds.append(wndId)
    return wndIds


# --- Converts a list of babelnet ids to a list of corresponding lexnames. ---
# :param bnIds: list of babelnet ids
# :param bn2lex_dict: dictionary with keys babelnet ids and values the corresponding lexname ones
# :return lexIds: corresponding list of lexnames ids

def convert_bn2lex(bnIds, bn2lex_dict):
    lexIds = []
    for elem in bnIds:
        lexId = bn2lex_dict[elem]
        lexIds.append(lexId)
    return lexIds


# ----- Methods for converting the gold file of the evaluation dataset to match the format required for all three tasks and compute the score. -----

# --- It converts the gold file for the babelnet prediction task. ---
# :param path_gold: path of the gold file of the dataset
# :param path_output: path of the output file where to put the converted version
# :return None

def convert_golds(path_gold, path_output):

    wn2bn_dict = get_dictionary('../resources/babelnet2wordnet.tsv', 1)
    in_file = open(path_gold, encoding='utf-8')
    out_file = open(path_output, 'w', encoding='utf-8')
    line = in_file.readline()
    while line != '':
        parts = line.split()
        instanceId = parts[0]
        wnId= sensekey2wn(parts[1])
        bnId = wn2bn_dict[wnId]
        out_file.write(instanceId + ' ' + bnId + '\n')
        line = in_file.readline()
    in_file.close()
    out_file.close()


# --- It converts the gold file for the wordnet domain prediction task. ---
# :param path_gold: path of the gold file of the dataset
# :param path_output: path of the output file where to put the converted version
# :return None

def convert_golds_wnd(path_gold, path_output):

    wn2bn_dict = get_dictionary('../resources/babelnet2wordnet.tsv', 1)
    bn2wnd_dict = get_dictionary('../resources/babelnet2wndomains.tsv', 0)
    in_file = open(path_gold, encoding='utf-8')
    out_file = open(path_output, 'w', encoding='utf-8')
    line = in_file.readline()
    while line != '':
        parts = line.split()
        instanceId = parts[0]
        wnId= sensekey2wn(parts[1])
        bnId = wn2bn_dict[wnId]
        try:
            wndId = bn2wnd_dict[bnId]
        except:
            wndId = 'factotum'
        out_file.write(instanceId + ' ' + wndId + '\n')
        line = in_file.readline()
    in_file.close()
    out_file.close()


# --- It converts the gold file for the lexnames prediction task. ---
# :param path_gold: path of the gold file of the dataset
# :param path_output: path of the output file where to put the converted version
# :return None

def convert_golds_lex(path_gold, path_output):

    wn2bn_dict = get_dictionary('../resources/babelnet2wordnet.tsv', 1)
    bn2lex_dict = get_dictionary('../resources/babelnet2lexnames.tsv', 0)
    in_file = open(path_gold, encoding='utf-8')
    out_file = open(path_output, 'w', encoding='utf-8')
    line = in_file.readline()
    while line != '':
        parts = line.split()
        instanceId = parts[0]
        wnId= sensekey2wn(parts[1])
        bnId = wn2bn_dict[wnId]
        lexId = bn2lex_dict[bnId]
        out_file.write(instanceId + ' ' + lexId + '\n')
        line = in_file.readline()
    in_file.close()
    out_file.close()
    