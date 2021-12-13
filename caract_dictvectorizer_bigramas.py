from lib import *
import os
import spacy
from sklearn.feature_extraction import DictVectorizer
import spacy_spanish_lemmatizer
from scipy import spatial

DIR = "/home/pc/Documents/textmining/dota/purged_es/"

files = os.listdir(DIR)
parrafo = ""

for filename in files:
    file = open(DIR + filename, "r")
    content = file.readlines()
    file.close()

    for l in range(len(content)):
        msg = content[l].split(",", maxsplit=3)[-1][:-1] + ". "
        parrafo += msg


doc, stopwords = nlp_process(corpus=parrafo)
words_frec = words_frecuency(doc=doc, stopwords=stopwords)


def maketriplas(doc, stopwords):
  corpus = {}

  for sent in doc.sents: # divido en oraciones
    sent_list = []
    for token in sent:
      if valid(word=token, stopwords=stopwords) and frec_threshold(token=token, words_frec=words_frec):
        tripla = token.pos_ + "_" + token.dep_ #+ "_" + next_word
        string = token2str(token)
        sent_list.append((string, tripla))

    for i in range(len(sent_list)-1):
      bigrama = sent_list[i][0] + "_" + sent_list[i+1][0]
      tripla_bigrama = sent_list[i][1] + "_" + sent_list[i+1][1]
      if bigrama not in corpus:
        corpus[bigrama] = {}
        corpus[bigrama][tripla_bigrama] = 1
      else:
        if tripla_bigrama in corpus[bigrama]:
          corpus[bigrama][tripla_bigrama] += 1
        else:
          corpus[bigrama][tripla_bigrama] = 1

  return corpus


vectorizer = DictVectorizer()
triplas = maketriplas(doc=doc, stopwords=stopwords)
vocabulary_dictvect, vocab_list, word_list = makevocab(corpus=triplas)
vectors_dictvect = vectorizer.fit_transform(vocab_list)

print(expandir_desescaladores(vocabulary=vocabulary_dictvect, vectors=vectors_dictvect, word_list=word_list))
