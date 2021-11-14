from sklearn.feature_extraction import DictVectorizer
import spacy_spanish_lemmatizer
import spacy
import os

from lib import *

files = os.listdir(DIR)
parrafo = get_only_messages(DIR=DIR, files=files)

def maketriplas(parrafo):
  corpus = {}

  doc, stopwords = nlp_process(corpus=parrafo)
  words_frec = words_frecuency(doc=doc, stopwords=stopwords)

  for sent in doc.sents: # divido en oraciones
    sent_list = []
    for token in sent:
      if valid(token, stopwords) and frec_threshold(token=token, words_frec=words_frec):         
        tripla = token.pos_ + "_" + token.dep_ #+ "_" + next_word
        string = token2str(token)
        sent_list.append((string, tripla))
    
    for i in range(len(sent_list)-2):
      trigrama = sent_list[i][0] + "_" + sent_list[i+1][0] + "_" + sent_list[i+2][0]
      tripla_trigrama = sent_list[i][1] + "_" + sent_list[i+1][1] + "_" + sent_list[i+2][1]
      if trigrama not in corpus:
        corpus[trigrama] = {}
        corpus[trigrama][tripla_trigrama] = 1
      else:
        if tripla_trigrama in corpus[trigrama]:
          corpus[trigrama][tripla_trigrama] += 1
        else:
          corpus[trigrama][tripla_trigrama] = 1

  return corpus

vectorizer = DictVectorizer()
triplas = maketriplas(parrafo=parrafo)
vocabulary_dictvect, vocab_list = makevocab(corpus=triplas)
vectors_dictvect = vectorizer.fit_transform(vocab_list)

# km_dictvect = run_kmeans(vectors=vectors_dictvect, normalize=True, clusters_number=3)
# show_clusters(vocabulary_dictvect, km_dictvect)

# plotted_points = plot(vectors=vectors_dictvect, vocabulary=vocabulary_dictvect)

# for p in plotted_points:
#         if len(plotted_points[p]) > 1:
#             print(p, plotted_points[p])

# for x in triplas:
#   print(x, triplas[x])

average_metrica(n=10, vectors=vectors_dictvect, normalize=False, clusters_number=n_clusters, vocab=vocabulary_dictvect)
average_metrica(n=10, vectors=vectors_dictvect, normalize=True, clusters_number=n_clusters, vocab=vocabulary_dictvect)
