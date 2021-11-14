from sklearn.feature_extraction import DictVectorizer
import spacy_spanish_lemmatizer
import spacy
import os

from lib import *

DIR = "/home/pc/Documents/textmining/dota/purged_es/"

files = os.listdir(DIR)
parrafo = get_only_messages(DIR=DIR, files=files)


def maketriplas(parrafo):
  corpus = {}

  nlp = spacy.load("es_core_news_sm")
  nlp.add_pipe("sentencizer")
  nlp.replace_pipe("lemmatizer", "spanish_lemmatizer")
  doc = nlp(parrafo)
  stopwords = nlp.Defaults.stop_words

  words_frec = words_frecuency(doc=doc, stopwords=stopwords)

  c = {}
  for x in words_frec:
    frec = words_frec[x]
    if frec > 5:
      frec = 5
    if frec in c:
      c[frec].append(x)
    else:
      c[frec] = [x]

  for sent in doc.sents: # divido en oraciones
    sent_list = []
    insulto = False
    toxic = False
    for token in sent:
      if valid(token, stopwords) and frec_threshold(token=token, words_frec=words_frec):         
        tripla = token.pos_ + "_" + token.dep_ #+ "_" + next_word
        string = token2str(token)
        if string in insultos:
          insulto = True

        if string in toxics:
          toxic = True

        sent_list.append((string, tripla, insulto, toxic))

        
    for i in range(1, len(sent_list)-1):
      trigrama = sent_list[i][0]
      prev_word = sent_list[i-1][0]
      pos_word = sent_list[i+1][0]
      insulto = bool2int(sent_list[i][2])
      toxic = bool2int(sent_list[i][3])
      tripla_trigrama = sent_list[i-1][1] + "_" + sent_list[i][1] + "_" + sent_list[i+1][1]
      if trigrama not in corpus:
        corpus[trigrama] = {"prev": [prev_word],
                            "pos": [pos_word],
                            "tripla": [tripla_trigrama],
                            "insulto": insulto,
                            "toxic": toxic}
      else:
        corpus[trigrama]["prev"].append(prev_word)
        corpus[trigrama]["pos"].append(pos_word)
        corpus[trigrama]["tripla"].append(tripla_trigrama)
        corpus[trigrama]["toxic"] = bool2int(insulto)
        corpus[trigrama]["insulto"] = bool2int(toxic)

  return corpus

vectorizer = DictVectorizer()
triplas = maketriplas(parrafo=parrafo)
vocabulary_dictvect, vocab_list = makevocab(corpus=triplas)
vectors_dictvect = vectorizer.fit_transform(vocab_list)

# km_dictvect = run_kmeans(vectors=vectors_dictvect, normalize=True, clusters_number=40)
# show_clusters(vocabulary_dictvect, km_dictvect)

# plotted_points = plot(vectors=vectors_dictvect, vocabulary=vocabulary_dictvect)

# for p in plotted_points:
  # if len(plotted_points[p]) > 1:
    # print(round(p[0], 2), round(p[1], 2), plotted_points[p])

# for x in triplas:
#   print(x, triplas[x])

average_metrica(n=10, vectors=vectors_dictvect, normalize=False, clusters_number=n_clusters, vocab=vocabulary_dictvect)
average_metrica(n=10, vectors=vectors_dictvect, normalize=True, clusters_number=n_clusters, vocab=vocabulary_dictvect)
