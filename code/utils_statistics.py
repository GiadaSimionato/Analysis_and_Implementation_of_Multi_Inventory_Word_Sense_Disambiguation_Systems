# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Script that analyse the features of the datasets.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains a method that plot the distribution of the lengths of each sentence of a specific dataset.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# --- Method that plot  the distribution of the lengths of each sentence of a specific dataset. ---
# :param sentences: list of sentences of a specific dataset
# :return list: return a list of all the relative frequencies for each unit of length

def get_lenSen(sentences):
    tot = 0
    length = np.zeros(259)
    for sentence in sentences:
        l = len(sentence)
        length[l]+=1
        tot +=1
    return length/tot


print('Starting OMSTI+SemCor analysis...')
sentences = np.load('OMSTI_inputs_BN.npy')
sentences = sentences[:100000]
length = get_lenSen(sentences)
plt.plot(length, 'r')
plt.xlabel('Sequence Length')
plt.ylabel('Relative Frequency')
plt.show()
print('Done')

print('Starting SemCor analysis...')
sentences = np.load('sentences.npy')
length = get_lenSen(sentences)
plt.plot(length, 'r')
plt.xlabel('Sequence Length')
plt.ylabel('Relative Frequency')
plt.show()
print('Done')