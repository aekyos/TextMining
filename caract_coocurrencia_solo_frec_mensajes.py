from lib import *

import numpy as np
import time
import vsm
import os


EXTRAS = 5
# 5. Mensajes en los ultimos 10 segs - 0 mensajes   A estos lo hago concatenando la lista
# 6                                    1 mensaje
# 7                                    2-4
# 8                                    5-10
# 9                                    mas de 10

files = os.listdir(DIR)
parrafo = get_messages_with_time(DIR=DIR, files=files)


def index_frec(n):
    if n == 0: return 0
    elif n == 1: return 1
    elif n >= 2 and n < 5: return 2
    elif n >= 5 and n < 10: return 3
    else: return 4


def index_sent_len(sent_len):
    if sent_len <= 10: return 0
    else: return 1


def count(sentence, counter, index, words_frec, stopwords, words_time, last_times, words_sent_len, words_toxic, words_insult):
    words_in_time_window = 0
    sent_length = len(sentence)
    toxic = False
    insult = False
    toxicity_checked = False
    sentence_purged = []
    for word1 in sentence:
        if word1.text[:4] == "TIME":
            time = float(word1.text[4:])
            for t in last_times:
                if time - t < 10:
                    words_in_time_window += 1
            last_times.append(time)
        else:
            if valid(word=word1, stopwords=stopwords) and frec_threshold(token=word1, words_frec=words_frec):
                sentence_purged.append(word1)
    # print("SP", sentence_purged)
    
    bigramas = []
    for word_index in range(len(sentence_purged)-1):
        word1 = sentence_purged[word_index]
        word2 = sentence_purged[word_index + 1]
        bigrama = token2str(token=word1) + "_" + token2str(token=word2)
        bigramas.append(bigrama)
    # print("B", bigramas)

    for word in bigramas:
        if word not in counter:
            counter[word] = {}
            index[word] = len(index)
        for coocurrencia in bigramas:
            if coocurrencia not in counter[word]:
                counter[word][coocurrencia] = 1
            else:
                counter[word][coocurrencia] += 1

        # update words_time dictionary
        index_frec_res = index_frec(n=words_in_time_window)
        if word not in words_time:
            words_time[word] = [0] * 5
        words_time[word][index_frec_res] += 1

def make_matrix(counter, index, words_time):
    matrix = []
    for word in counter:
        row = [0]*(len(counter) + EXTRAS)
        matrix.append(row)

    for word in counter:
        x = index[word]
        for coocurrencia in counter[word]:
            y = index[coocurrencia]
            
            value = counter[word][coocurrencia]
            matrix[x][y] = value

        i = len(matrix[x]) - EXTRAS
        for elem in words_time[word]:
            matrix[x][i] = elem
            i += 1

    return matrix

counter = {}
index = {}

doc, stopwords = nlp_process(corpus=parrafo)

words_frec = words_frecuency(doc=doc, stopwords=stopwords)
times = []
words_time = {}
words_sent_len = {}
words_toxic = {}
words_insult = {}
last_times = []
for sent in doc.sents: # divido en oraciones
    count(sentence=sent, counter=counter, index=index, words_frec=words_frec, stopwords=stopwords, last_times=last_times, words_time=words_time, words_sent_len=words_sent_len, words_toxic=words_toxic, words_insult=words_insult)
m = make_matrix(counter=counter, index=index, words_time=words_time)
average_metrica(n=5, vectors=m, normalize=True, clusters_number=n_clusters, vocab=counter)