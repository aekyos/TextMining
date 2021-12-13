from gensim.models import Word2Vec
from lib import *

import os
import spacy
import spacy_spanish_lemmatizer
import numpy
import gensim


files = os.listdir(DIR)
parrafo = get_messages_with_time(DIR=DIR, files=files)

nlp = spacy.load("es_core_news_sm")
nlp.add_pipe("sentencizer")
nlp.replace_pipe("lemmatizer", "spanish_lemmatizer")
doc = nlp(parrafo[:1000000])
stopwords = nlp.Defaults.stop_words

words_frec = words_frecuency(doc=doc, stopwords=stopwords)


def tokenize_sentence(sentence, words_frec, stopwords, tokenized):
    for word1 in sentence:
        if valid(word=word1, stopwords=stopwords):
            word = token2str(token=word1)
            tokenized.add(word)


def tokenize(doc, words_frec, stopwords=stopwords):
    tokenized = set()
    for sent in doc.sents:
        tokenize_sentence(sentence=sent, words_frec=words_frec, stopwords=stopwords, tokenized=tokenized)
    return tokenized


def gen_vectors(normalized):
    vocab = list(normalized)
    model = gensim.models.KeyedVectors.load_word2vec_format('SBW-vectors-300-min5-cbow.txt', binary=False)
    matrix = [] # model[w] for w in vocab]

    for w in vocab:
        if w in model: # si las palabras estan en el modelo entrenado
            matrix.append(model[w])

    return vocab, matrix


tokenized = tokenize(doc=doc, words_frec=words_frec, stopwords=stopwords)
print(tokenized)
vocab, vectors = gen_vectors(tokenized)

average_metrica(n=10, vectors=vectors, normalize=True, clusters_number=n_clusters, vocab=vocab)
