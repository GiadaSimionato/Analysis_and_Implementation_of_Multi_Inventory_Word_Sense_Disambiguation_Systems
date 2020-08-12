# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Script that contains the methods to parse the XML file of the dataset and extract useful information.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It parses the dataset to extract information such as input sentences, POS and ids, even in a blind way.
# It additionally creates the labels in such a way to map all the wf tags' content to a standard keyword.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

from lxml import etree
from utils_mappings import inst2bn, inst2cg


# --- Method that parses the XML file in the Raganato et al. format extracting input, POS and ids. --- 
# :param xml_path: path of the XML file to parse
# :param inst2sk_dict: dictionary whose keys are the instance ids and the values are the sensekeys
# :param wn2bn_dict: dictionary with keys wordnet ids and values the corresponding babelnet ones
# :param bn2cg_dict: dictionary with keys the babelnet ids and values the coarse-grained ids (wordnet domains or lexnames)
# :param control: string to identify the task and collect labels directly with the ids in the corresponding format
# :return inputs: list of lists of lemmas
# :return labels: list of lists of labels, if the words need to be disambiguated then collect the corresponding ids
# :return Ids: list of ids encountered in the file

def parse(xml_path, inst2sk_dict, wn2bn_dict, bn2cg_dict, control):

    inputs = []
    labels = []
    Ids = []
    content = etree.iterparse(xml_path, remove_blank_text=True, tag = 'sentence')
    for event, element in content:
        inp = []
        label = []
        for child in element:
            #inp.append(child.text.lower())
            inp.append(child.get('lemma').lower())
            if child.tag == 'wf':
                #label.append(child.text.lower())
                label.append(child.get('lemma').lower())
            elif child.tag == 'instance':
                if control=='BN':
                    Id = inst2bn(child.get('id'), inst2sk_dict, wn2bn_dict)
                elif control=='WND':
                    Id = inst2cg(child.get('id'), inst2sk_dict, wn2bn_dict, bn2cg_dict, control)
                elif control=='LN':
                    Id = inst2cg(child.get('id'), inst2sk_dict, wn2bn_dict, bn2cg_dict, control)
                label.append(Id)
                Ids.append(Id)
        inputs.append(inp)
        labels.append(label)
        element.clear()
    return inputs, labels, Ids


# --- Methods to extract information from a XML file in a different format (for the evaluation dataset). ---
# :param xml_path: path of the XML file to parse
# :return inputs: list of lists of input sentences
# :return pos: list of lists of corresponding POS
# :return flags: list of lists of boolean values: True if the word must be disambiguated, False otherwise
# :return ids: list of lists of ids: '' if the word mustn't be disambiguated, id otherwise

def blind_parsing(xml_path):

    inputs = []
    pos = []
    flags = []
    ids = []

    content = etree.iterparse(xml_path, remove_blank_text=True, tag = 'sentence')
    for event, element in content:
        partials = []
        partial_pos = []
        partial_flags = []
        partial_ids = []
        for child in element:
            partials.append(child.get('lemma'))
            partial_pos.append(child.get('pos'))
            if child.tag == 'wf':
                partial_flags.append(False)
                partial_ids.append('')
            if child.tag == 'instance':
                partial_flags.append(True)
                partial_ids.append(child.get('id'))
        inputs.append(partials)
        pos.append(partial_pos)
        flags.append(partial_flags)
        ids.append(partial_ids)
        element.clear()
    return inputs, pos, flags, ids


# --- Method that extracts the labels in a restricted way: all the words that don't need to be disambiguated are mapped to the keyword '<LEMMA>'. ---
# :param path_xml: path of the XML file to parse
# :param inst2sk_dict: dictionary whose keys are the instance ids and the values are the sensekeys
# :param wn2bn_dict: dictionary with keys wordnet ids and values the corresponding babelnet ones
# :param bn2cg_dict: dictionary with keys the babelnet ids and values the coarse-grained ids (wordnet domains or lexnames)
# :param control: string to identify the task and collect labels directly with the ids in the corresponding format
# :return labels: list of lists of elements: '<LEMMA>' if the word is not to be disamb. or the corresponding format of the ids otherwise

def get_restriced_labels(path_xml, inst2sk_dict, wn2bn_dict, bn2cg_dict, control):

    labels = []
    content = etree.iterparse(path_xml, remove_blank_text=True, tag = 'sentence')
    for event, element in content:
        label = []
        for child in element:
            if child.tag == 'wf':       # if it must be disamb.
                label.append('<LEMMA>')
            elif child.tag == 'instance':   # otherwise
                Id = ''
                if control=='BN':       # for fine-grained application
                    Id = inst2bn(child.get('id'), inst2sk_dict, wn2bn_dict)
                elif control=='WND':    # for wordnet domains application
                    Id = inst2cg(child.get('id'), inst2sk_dict, wn2bn_dict, bn2cg_dict, control)
                elif control=='LN':     # for lexnames application
                    Id = inst2cg(child.get('id'), inst2sk_dict, wn2bn_dict, bn2cg_dict, control)
                label.append(Id)
        labels.append(label)
    return labels