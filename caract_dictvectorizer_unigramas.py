from sklearn.feature_extraction import DictVectorizer
from lib import *
import spacy_spanish_lemmatizer
import spacy
import os


files = os.listdir(DIR)
parrafo = ""

for filename in files:
    file = open(DIR + filename, "r")
    content = file.readlines()
    file.close()

    for l in range(len(content)):
        msg = content[l].split(",", maxsplit=3)[-1][:-1] + ". "
        parrafo += msg

nlp = spacy.load("es_core_news_sm")
nlp.add_pipe("sentencizer")
nlp.replace_pipe("lemmatizer", "spanish_lemmatizer")
doc = nlp(parrafo)

stopwords = nlp.Defaults.stop_words

words_frec = words_frecuency(doc=doc, stopwords=stopwords)


def maketriplas(doc, stopwords):
  corpus = {}

  for token in doc: # divido en oraciones
    if valid(word=token, stopwords=stopwords) and frec_threshold(token=token, words_frec=words_frec):
      try:
        next_word = token2str(doc[token.i+1])
      except:
        next_word = ""
      tripla = token.pos_ + "_" + token.dep_ #+ "_" + next_word
      string = token2str(token)
      if string not in corpus:
        corpus[string] = {}
        corpus[string][tripla] = 1
      else:
        if tripla in corpus[string]:
          corpus[string][tripla] += 1
        else:
          corpus[string][tripla] = 1

  return corpus


vectorizer = DictVectorizer()
triplas = maketriplas(doc=doc, stopwords=stopwords)
vocabulary_dictvect, vocab_list = makevocab(corpus=triplas)
vectors_dictvect = vectorizer.fit_transform(vocab_list)

# km_dictvect = run_kmeans(vectors=vectors_dictvect, normalize=True, clusters_number=3)
# show_clusters(vocabulary_dictvect, km_dictvect)

# print(vocabulary_dictvect)
# plot()

average_metrica(n=10, vectors=vectors_dictvect, normalize=False, clusters_number=n_clusters, vocab=vocabulary_dictvect)
average_metrica(n=10, vectors=vectors_dictvect, normalize=True, clusters_number=n_clusters, vocab=vocabulary_dictvect)
