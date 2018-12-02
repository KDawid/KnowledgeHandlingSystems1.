import json
from enum import Enum
from gensim import corpora
import spacy
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
import math
import numpy as np
import pandas as pd
from pprint import pprint  # pretty-printer
from unidecode import unidecode
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

STOPLIST = set('for a of the and to in'.split())
MIN_FREQUENCY = 5
MAX_FREQUENCY = 100

class PREPROCESS_TYPE(Enum):
    GENSIM = "gensim"
    SPACY = "spacy"
    #IMPLEMENTED = 2

class TRANSFORMATION_TYPE(Enum):
    TF_IDF = "tf-idf"
    WORD2VEC = "word2vec"
    TFIDF_W2V = "w2v-tfidf"

class FakeNewsPreprocesser:
    def __init__(self, configFilePath):
        with open(configFilePath) as json_data_file:
            config = json.load(json_data_file)
        self.JSON_FILE_PATH = config["JSON_FILE_PATH"]
        self.DICTIONARY_FILE_PATH = config["DICTIONARY_FILE_PATH"]
        self.SAVE_WORDS_FILE_PATH = config["SAVE_WORDS_FILE_PATH"]
        self.CORPUS_FILE_PATH = config["CORPUS_FILE_PATH"]
        self.RESULT_FILE_PATH = config["RESULT_FILE_PATH"]
        self.WORD2VEC_MODEL_FILE_PATH = config["WORD2VEC_MODEL_FILE_PATH"]

    def readJson(self):
        return pd.read_json(self.JSON_FILE_PATH)

    def preprocessText(self, jsonData, type):
        texts = jsonData['content']
        if type == PREPROCESS_TYPE.GENSIM:
            texts = [
                text.replace("“", " ")
                    .replace("…", " ")
                    .replace(",", " ")
                    .replace("‘", " ")
                    .replace("”", " ")
                    .replace("’", " ")
                    .replace("—", " ")
                    .replace("-", " ")
                    .replace("  ", " ") for text in jsonData['content']]
            texts = [preprocess_string(text) for text in texts]
        elif type == PREPROCESS_TYPE.SPACY:
            self.spacy_nlp = spacy.load('en_core_web_sm')
            texts = self.callSpacy(jsonData)
        return texts

    def implementedPreprocessing(self, data):
        texts = [[word.lower() for word in document if word.lower() not in STOPLIST] for document in data]
        texts = self.applyMinFrequency(texts, minFrequency=MIN_FREQUENCY, maxFrequency=MAX_FREQUENCY)
        return texts

    def applyMinFrequency(self, texts, minFrequency=1, maxFrequency=math.inf):
        # remove words that appear only once
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
              frequency[token] += 1
        import operator
        return [[token for token in text if minFrequency <= frequency[token] <= maxFrequency] for text in texts]

    def preprocessWithSpacy(self, text):
        doc = self.spacy_nlp(text)
        stopwords = spacy.lang.en.STOP_WORDS
        tokens = []
        for token in doc:
            if token.text not in stopwords and \
                    not token.text.isnumeric() and \
                    not token.is_punct and \
                    not token.is_digit and \
                    not token.is_space and \
                    not token.like_url and \
                    len(token) > 1 and \
                    "," not in token.text and \
                    "’" not in token.text:
                tokens.append(token.text)

        tokens = [token.lower() for token in tokens]
        return tokens

    def detectBigrams(self, text):
        from gensim.models.phrases import Phrases, Phraser
        phrases = Phrases(text, min_count=20)
        bigram = Phraser(phrases)
        for token in bigram[text]:
            if '_' in token:
                text.append(token)
        return text

    def callSpacy(self, jsonData):
        texts = jsonData['content']
        texts = [self.preprocessWithSpacy(text) for text in texts]
        texts = [self.detectBigrams(text) for text in texts]
        return texts

    def printDictionaryItems(self, dictionary):
        for (id, word) in dictionary.items():
            print(id, word)

    def saveDictionaryWords(self, dictionary):
        wordSet = set()
        for (id, word) in dictionary.items():
            wordSet.add(word)
        with open(self.SAVE_WORDS_FILE_PATH, 'w') as f:
            for word in sorted(wordSet):
                f.write(unidecode(word) + "\n")

    def saveDictionary(self, dictionary):
        with open(self.DICTIONARY_FILE_PATH, 'w') as f:
            for (id, word) in dictionary.items():
                f.write(str(id) + "," + unidecode(word) + "\n")

    def saveTfIdfCorpus(self, model, corpus):
        with open(self.CORPUS_FILE_PATH, 'w') as f:
            f.write("[")
            for i in range(len(corpus)):
                f.write(json.dumps(model[corpus[i]]))
                if i != len(corpus) - 1:
                    f.write(",")
            f.write("]")

    def getNumberOfVectors(self):
        dictionary = pd.read_csv(self.DICTIONARY_FILE_PATH)
        return len(dictionary)

    def writeTfIdfWordVectorsToJson(self, jsonData, model, corpus, filePath):
        vectorLength = self.getNumberOfVectors()
        tfCorpus = [model[corpus[i]] for i in range(len(corpus))]
        tfIdf = dict()
        tfIdf['type'] = jsonData['type']

        d = dict()
        d['type'] = [tfIdf['type'][0]]
        for i in range(vectorLength):
            if i in [key for (key, value) in tfCorpus[0]]:
                d[str(i)] = [value for (key, value) in tfCorpus[0] if key == i]
            else:
                d[str(i)] = [0.0]
        for index in range(1, len(tfIdf['type'])):
            d['type'] += [tfIdf['type'][index]]
            for i in range(vectorLength):
                if i in [key for (key, value) in tfCorpus[index]]:
                    d[str(i)] += [value for (key, value) in tfCorpus[index] if key == i]
                else:
                    d[str(i)] += [0.0]

        with open(filePath, 'w') as f:
            out = json.dumps(d, indent=4)
            f.write(out)

    def tfIdfTransformation(self, texts, jsonData, filePath):
        #texts = texts.values.tolist()
        #texts = [text.split(" ") for text in texts]
        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in texts]
        model = TfidfModel(corpus)  # fit model

        self.saveDictionaryWords(dictionary)
        self.saveTfIdfCorpus(model, corpus)
        self.saveDictionary(dictionary)
        self.writeTfIdfWordVectorsToJson(jsonData, model, corpus, filePath)

    # source: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    def word2VecTransformation(self, texts, data, filePath):
        model = KeyedVectors.load_word2vec_format(self.WORD2VEC_MODEL_FILE_PATH, binary=True)
        word2vec = dict(zip(model.wv.index2word, model.wv.syn0))
        dim = len(next(iter(word2vec.items())))
        result = np.array([
            np.mean([model[w] for w in words if w in model]
                    or [np.zeros(dim)], axis=0)
            for words in texts
        ])
        df = pd.DataFrame(result, index=range(len(result)))
        df['type'] = data['type']
        result = df.to_dict()
        with open(filePath, 'w') as f:
            out = json.dumps(result, indent=4)
            f.write(out)

    def word2VecTfIdfTransformation(self, texts, data, filePath):
        model = KeyedVectors.load_word2vec_format(self.WORD2VEC_MODEL_FILE_PATH, binary=True)
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))

        dim = len(next(iter(w2v.items())))

        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(texts)

        max_idf = max(tfidf.idf_)
        word2weight = defaultdict(
            lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        result = np.array([
            np.mean([w2v[w] * word2weight[w]
                     for w in words if w in w2v] or
                    [np.zeros(dim)], axis=0)
            for words in texts
        ])
        df = pd.DataFrame(result, index=range(len(result)))
        df['type'] = data['type']
        result = df.to_dict()
        with open(filePath, 'w') as f:
            out = json.dumps(result, indent=4)
            f.write(out)

    def transformData(self, texts, jsonData, type, filePath):
        if type == TRANSFORMATION_TYPE.TF_IDF:
            self.tfIdfTransformation(texts, jsonData, filePath)
        elif type == TRANSFORMATION_TYPE.WORD2VEC:
            self.word2VecTransformation(texts, jsonData, filePath)
        elif type == TRANSFORMATION_TYPE.TFIDF_W2V:
            self.word2VecTfIdfTransformation(texts, jsonData, filePath)
