from lib import *

import numpy as np
import time
import vsm
import os


EXTRAS = 10
# 1. Insultos
# 2. Toxic
# 3. Largo oracion(0-10)
# 4. Largo oracion(10>)
# 5. Mensajes en los ultimos 10 segs - 0 mensajes   A estos lo hago concatenando la lista
# 6                                    1 mensaje
# 7                                    2-4
# 8                                    5-10
# 9                                    mas de 10
# 10. Mensajes en los ultimso 20 segs ??

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
    for word1 in sentence:
        if word1.text[:4] == "TIME":
            time = float(word1.text[4:])
            for t in last_times:
                if time - t < 10:
                    words_in_time_window += 1
            last_times.append(time)
        else:
            if valid(word=word1, stopwords=stopwords) and frec_threshold(token=word1, words_frec=words_frec):
                word = token2str(token=word1)
                if word not in counter:
                    counter[word] = {}
                    index[word] = len(index)
                for word2 in sentence:
                    if valid(word=word2, stopwords=stopwords) and frec_threshold(token=word2, words_frec=words_frec):
                        coocurrencia = token2str(token=word2)
                        if coocurrencia not in counter[word]:
                            counter[word][coocurrencia] = 1
                        else:
                            counter[word][coocurrencia] += 1

                        # Check toxicity
                        if not toxicity_checked:
                            if coocurrencia in toxics:
                                toxic = True

                        # Check insults
                        if not toxicity_checked:
                            if coocurrencia in insultos:
                                insult = True
                toxicity_checked = True

                # update words_toxic dictionary
                if word not in words_toxic:
                    words_toxic[word] = bool2int(toxic)
                else:
                    words_toxic[word] += bool2int(toxic)

                # update words_insult dictionary
                if word not in words_insult:
                    words_insult[word] = bool2int(insult)
                else:
                    words_insult[word] += bool2int(insult)

                # update words_time dictionary
                index_frec_res = index_frec(n=words_in_time_window)
                if word not in words_time:
                    words_time[word] = [0] * 5
                words_time[word][index_frec_res] += 1

                # update words_sent_len
                index_sent_len_res = index_sent_len(sent_len=sent_length)
                if word not in words_sent_len:
                    words_sent_len[word] = [0] * 2
                words_sent_len[word][index_sent_len_res] += 1


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
        matrix[x][i] = 5 # insulto
        matrix[x][i] = 6 # toxic

        i += 2
        for elem in words_sent_len[word]:
            matrix[x][i] = elem
            i += 1

        for elem in words_time[word]:
            matrix[x][i] = elem
            i += 1

    return matrix


counter = {}
index = {}

t1 = time.time()
doc, stopwords = nlp_process(corpus=parrafo)
t2 = time.time()
print("nlp_process done", t2-t1)

words_frec = words_frecuency(doc=doc, stopwords=stopwords)
t3 = time.time()
print("words_frecuency done", t3-t2)
times = []
words_time = {}
words_sent_len = {}
words_toxic = {}
words_insult = {}
last_times = []
for sent in doc.sents: # divido en oraciones
    count(sentence=sent, counter=counter, index=index, words_frec=words_frec, stopwords=stopwords, last_times=last_times, words_time=words_time, words_sent_len=words_sent_len, words_toxic=words_toxic, words_insult=words_insult)
t4 = time.time()
print("count done", t4-t3)
m = make_matrix(counter=counter, index=index, words_time=words_time)
t5 = time.time()
print("make_matrix done", t5-t4)
average_metrica(n=1, vectors=m, normalize=True, clusters_number=n_clusters, vocab=counter)
t6 = time.time()
print("average_metrica 1 done", t6-t5)
average_metrica(n=4, vectors=m, normalize=False, clusters_number=n_clusters, vocab=counter)
t7 = time.time()
print("average_metrica 2 done", t7-t6)

# vectors_km = run_kmeans(vectors=m, normalize=False, clusters_number=50)
# metrica_clusters(vocabulary=counter, model=vectors_km, desescaladores=desescaladores)
# show_clusters(counter, vectors_km)

# vectors_df = pd.DataFrame(np.array(m, dtype='float64'), index=list(index.keys()))
# print(vsm.neighbors('callate', vectors_df).head())

# vectors_df_lsa = vsm.lsa(vectors_df, k=500)
# plotted_points = plot(vectors=vectors_df_lsa, vocabulary=index)
# vectors_lsa_km = run_kmeans(vectors=vectors_df_lsa, normalize=False, clusters_number=50)
# show_clusters(counter, vectors_lsa_km)

# vectors_df_lsa = vsm.lsa(vectors_df, k=20)
# plotted_points = plot(vectors=vectors_df_lsa, vocabulary=index)

# vectors_df_lsa = vsm.lsa(vectors_df, k=10)
# plotted_points = plot(vectors=vectors_df_lsa, vocabulary=index)

# for x in words_frec:
#     print(x, words_frec[x])