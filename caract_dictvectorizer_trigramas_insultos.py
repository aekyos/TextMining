from sklearn.feature_extraction import DictVectorizer
import spacy_spanish_lemmatizer
import spacy
import os

from lib import *

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
      trigrama = sent_list[i-1][0] + "_" + sent_list[i][0] + "_" + sent_list[i+1][0]
      tripla_trigrama = sent_list[i-1][1] + "_" + sent_list[i][1] + "_" + sent_list[i+1][1]
      insulto = sent_list[i][2]
      toxic = sent_list[i][3]
      if trigrama not in corpus:
        corpus[trigrama] = {}
        corpus[trigrama][tripla_trigrama] = 1
        corpus[trigrama]["insulto"] = bool2int(insulto)
        corpus[trigrama]["toxic"] = bool2int(toxic)
      else:
        if tripla_trigrama in corpus[trigrama]:
          corpus[trigrama][tripla_trigrama] += 1
          corpus[trigrama]["insulto"] += bool2int(insulto)
          corpus[trigrama]["toxic"] = bool2int(toxic)
        else:
          corpus[trigrama][tripla_trigrama] = 1
          corpus[trigrama]["insulto"] = bool2int(insulto)
          corpus[trigrama]["toxic"] = bool2int(toxic)

  return corpus

vectorizer = DictVectorizer()
triplas = maketriplas(parrafo=parrafo)
vocabulary_dictvect, vocab_list = makevocab(corpus=triplas)
vectors_dictvect = vectorizer.fit_transform(vocab_list)

# km_dictvect = run_kmeans(vectors=vectors_dictvect, normalize=True, clusters_number=20)
# show_clusters(vocabulary_dictvect, km_dictvect)

# plotted_points = plot(vectors=vectors_dictvect, vocabulary=vocabulary_dictvect)

# for p in plotted_points:
  # if len(plotted_points[p]) > 1:
    # print(round(p[0], 2), round(p[1], 2), plotted_points[p])

# for x in triplas:
#   print(x, triplas[x])

average_metrica(n=10, vectors=vectors_dictvect, normalize=False, clusters_number=n_clusters, vocab=vocabulary_dictvect)
average_metrica(n=10, vectors=vectors_dictvect, normalize=True, clusters_number=n_clusters, vocab=vocabulary_dictvect)
