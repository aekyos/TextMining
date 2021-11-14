from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import Counter

import spacy_spanish_lemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import re

n_clusters = 40

desescaladores = ["tranquilo",
                  "callate", "calla", "callar", "callarse", #"calladito" ??
                  "terminenla", "terminala", #"terminen", "termina"
                  "muteado", "muteen", "muteo", #"mute", "muteame" ??
                 ]

insultos = {"cagada", "puta", "noob", "mierda", "asco", "verga", "peruano", 
            "puto", "perra", "negro", "virgen", "basura", "ctmre", "mono", 
            "rata", "concha", "chucha"}

toxics = {"troll", "report", "caca", "gg", "afk", "feed", "fedea", "lloron",
          "izi", "malo", "pena", "manco", "bot"}

DIR = "/home/pc/Documents/textmining/dota/purged_es/"


def nlp_process(corpus):
  nlp = spacy.load("es_core_news_sm")
  nlp.add_pipe("sentencizer")
  nlp.replace_pipe("lemmatizer", "spanish_lemmatizer")
  nlp.max_length = 2000000
  # corpus = corpus[:300000]
  doc = nlp(corpus, disable = ['ner', 'parser'])
  stopwords = nlp.Defaults.stop_words

  return doc, stopwords


def get_only_messages(DIR, files):
  parrafo = ""

  for filename in files:
      file = open(DIR + filename, "r")
      content = file.readlines()
      file.close()

      for l in range(len(content)):
          msg = content[l].split(",", maxsplit=3)[-1][:-1] + ". "
          parrafo += msg

  return parrafo


def get_messages_with_time(DIR, files):
  parrafo = ""

  for filename in files:
      file = open(DIR + filename, "r")
      content = file.readlines()
      file.close()

      for l in range(len(content)):
          parsed = content[l].split(",", maxsplit=3)#[-1][:-1] + ". "
          time = "TIME" + parsed[1] + " "
          msg = parsed[-1][:-1] + ". "
          parrafo += time + msg

  return parrafo


def token2str(token):
    string = token.lemma_
    string = string.lower()

    if len(string) > 3 and string[-1] == string[-2] and string[-2] == string[-3]:
        pattern = string[-1] + '{3,}'
        string = re.sub(pattern, string[-1], string)

    dict = {"dx": "xd", "xdd": "xd", "mrd": "mierda", "cacas": "caca", "mierdas": "mierda",
            "conchatuamre": "ctm", "conchatumare": "ctm", "mrda": "mierda", "reportenlo": "report",
            "ty": "gracias", "shh": "callate", "xdxd": "xd", "parao": "parado", "pliz": "please",
            "plis": "pls", "cagao": "cagado", "virjen": "virgen", "meirda": "mierda", "csmr": "ctm",
            "ctmr": "ctm", "reprot": "report", "plz": "pls", "tyy": "gracias", "trol": "troll",
            "tranqui": "tranquilo"}
    if string in dict:
        string = dict[string]

    return string


def run_kmeans(vectors, normalize, clusters_number):
    if normalize:
        vectors = preprocessing.normalize(vectors)

    km_model = KMeans(n_clusters=clusters_number)
    km_model.fit(vectors)

    return km_model


def retrieve_clusters(vocabulary, model):
  c = Counter(sorted(model.labels_))

  keysVocab = vocabulary
  if type(vocabulary) != type([]):
    keysVocab = list(vocabulary.keys())

  n = 0
  clusters = []
  for cluster in c:
    cluster_size = c[cluster]
    cluster = set()
    if cluster_size >= 2:
      word_indexs = [i for i,x in enumerate(list(model.labels_)) if x == n]
      for i in word_indexs:
        cluster.add(keysVocab[i])
    clusters.append(cluster)
    n += 1

  return clusters


def show_clusters(vocabulary, model):
  clusters = retrieve_clusters(vocabulary=vocabulary, model=model)

  n = 0
  for cluster in clusters:
    cluster_size = len(cluster)
    if cluster_size >= 0:
      print("Cluster %d: %d words" % (n, cluster_size))
      if cluster_size >= 1 and cluster_size <= 2500:
        print("Words:", end="")
        for word in cluster:
          print(' %s' % word, end=',')
        print("\n")
    n += 1

  print()


def show_clusters_bigramas(vocabulary, model):
  clusters = retrieve_clusters(vocabulary=vocabulary, model=model)

  n = 0
  for cluster in clusters:
    cluster_words = set()
    for bigrama in cluster:
      words = bigrama.split("_")
      for word in words:
        cluster_words.add(word)


    cluster_size = len(cluster_words)
    print("Cluster %d: %d words" % (n, cluster_size))
    # if cluster_size >= 0:
      # if cluster_size >= 1 and cluster_size <= 2500:
    print("Words:", end="")
    for word in cluster_words:
      print(' %s' % word, end=',')
    print("\n")
    n += 1

  print()


def metrica_clusters(vocabulary, model, desescaladores, output=False):
  clusters = retrieve_clusters(vocabulary=vocabulary, model=model)

  n = 0
  res = {}
  for cluster in clusters:
    desescaladores_in_cluster = []
    for word in desescaladores:
      if len(cluster) > 0 and "_" in next(iter(cluster)): # Si tiene esto es porque es un bigrama o un trigrama
        for c in cluster:
          if word in c:
            desescaladores_in_cluster.append(word)
      else: # es un unigrama
        if word in cluster:
          desescaladores_in_cluster.append(word)
    if len(desescaladores_in_cluster) > 1:
      res[n] = (len(cluster), desescaladores_in_cluster)

    n += 1

  metrica = 0
  if output:
    print("index | size | metrica | desescaladores in cluster")
  for cluster_index in res:
    cluster_size = res[cluster_index][0]
    deses_in_cluster = res[cluster_index][1]
    metrica_cluster = len(deses_in_cluster) / cluster_size
    if output:
      print("% 5d | % 4d | %0.5f |"%(cluster_index, cluster_size, metrica_cluster), deses_in_cluster)
    metrica += metrica_cluster
  if output:
    print("Metrica total = %f, con %d clusters."%(metrica, len(clusters)))
  return metrica


def average_metrica(n, vectors, normalize, clusters_number, vocab):
  acumulador = 0
  for _ in range(n):
    vectors_km = run_kmeans(vectors=vectors, normalize=normalize, clusters_number=clusters_number)
    res = metrica_clusters(vocabulary=vocab, model=vectors_km, desescaladores=desescaladores)
    acumulador += res
  avg = acumulador/n
  print("El promedio de la metrica con %d iteraciones y %d clusters es %f."%(n, clusters_number, avg))


def bool2int(b):
  if b:
    return 1
  else:
    return -1


def valid(word, stopwords):
  is_stop = word.text not in stopwords
  long_enough = len(word) > 1
  return not word.is_punct and long_enough and word.is_alpha and is_stop


def frec_threshold(token, words_frec): # saca las palabras que aparecen muy poco
  min = 1

  return words_frec[token2str(token)] > min


def words_frecuency(doc, stopwords):
  words_frec = {}

  for token in doc:
    if valid(word=token, stopwords=stopwords):
      word = token2str(token)
      if word not in words_frec:
        words_frec[word] = 1
      else:
        words_frec[word] += 1

  return words_frec


def makevocab(corpus):
    corpus_index = {}
    corpus_list = []

    count = 0
    for string in corpus:
        corpus_index[string] = count
        corpus_list.append(corpus[string])
        count += 1

    return corpus_index, corpus_list


def plot(vectors, vocabulary):
    # if type(vectors) != type([]):
        # vectors = vectors.toarray()
    X = pd.DataFrame(vectors)
    X_norm = (X - X.min())/(X.max() - X.min())

    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm))

    c = 0
    random_list = {"aa"}#"report", "puto", "rubick", "dame", "ez"}#"ez", "gg", "GG"}
    plotted_points = {}
    for word in vocabulary:
        if word not in random_list:
            x = transformed[0][c]
            y = transformed[1][c]
            plt.scatter(x, y, label=word)
            plt.annotate(word, (x, y))
            if (x,y) in plotted_points:
                plotted_points[(x,y)].append(word)
            else:
                plotted_points[(x,y)] = [word]
            c += 1

    plt.show()
    # plt.savefig("results.png")

    return plotted_points