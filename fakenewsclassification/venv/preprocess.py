import json
from gensim import corpora
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
from pprint import pprint  # pretty-printer
from unidecode import unidecode

CONFIG_FILE_PATH = "config.json"

STOPLIST = set('for a of the and to in'.split())
MIN_FREQUENCY = 5

class FakeNewsPreprocesser:
    JSON_FILE_PATH = None
    DICTIONARY_FILE_PATH = None
    SAVE_WORDS_FILE_PATH = None
    CORPUS_FILE_PATH = None
    RESULT_FILE_PATH = None
    VECTOR_FILE_PATH = None

    def __init__(self, configFilePath):
        with open(configFilePath) as json_data_file:
            config = json.load(json_data_file)
        self.JSON_FILE_PATH = config["JSON_FILE_PATH"]
        self.DICTIONARY_FILE_PATH = config["DICTIONARY_FILE_PATH"]
        self.SAVE_WORDS_FILE_PATH = config["SAVE_WORDS_FILE_PATH"]
        self.CORPUS_FILE_PATH = config["CORPUS_FILE_PATH"]
        self.RESULT_FILE_PATH = config["RESULT_FILE_PATH"]
        self.VECTOR_FILE_PATH = config["VECTOR_FILE_PATH"]

    def readJson(self):
        return pd.read_json(self.JSON_FILE_PATH)

    def preprocessText(self, jsonData):
        texts = jsonData['content']
        texts = [
            text.replace("“", " ")
                .replace("…", " ")
                .replace("‘", " ")
                .replace("”", " ")
                .replace("’", " ")
                .replace("—", " ")
                .replace("-", " ")
                .replace("  ", " ") for text in jsonData['content']]
        texts = [preprocess_string(text) for text in texts]
        return texts

    def implementedPreprocessing(self, data):
        texts = [[word for word in document.lower().split() if word not in STOPLIST] for document in data]
        texts = applyMinFrequency(texts, MIN_FREQUENCY)
        return text

    def applyMinFrequency(self, texts, i):
        # remove words that appear only once
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
              frequency[token] += 1
        return [[token for token in text if frequency[token] > i] for text in texts]

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

    def saveDataWithTfIdfInformation(self, model, corpus):
        tfCorpus = [model[corpus[i]] for i in range(len(corpus))]
        result = json.load(open(self.JSON_FILE_PATH))
        for i in range(len(tfCorpus)):
            result[i]["TF-IDF"] = dict(tfCorpus[i])
        with open(self.RESULT_FILE_PATH, 'w') as f:
            out = json.dumps(result, indent=4)
            f.write(out)

    def getNumberOfVectors(self):
        dictionary = pd.read_csv(self.DICTIONARY_FILE_PATH)
        return len(dictionary)

    def writeWordVectorsToJson(self):
        jsonData = pd.read_json(self.RESULT_FILE_PATH)
        vectorLength = self.getNumberOfVectors()

        tfIdf = jsonData['TF-IDF'].to_frame()
        tfIdf['type'] = jsonData['type']

        d = dict()
        d['type'] = [tfIdf['type'][0]]
        for i in range(vectorLength):
            if str(i) in tfIdf["TF-IDF"][0]:
                d[str(i)] = [tfIdf["TF-IDF"][0][str(i)]]
            else:
                d[str(i)] = [0.0]
        for index in range(len(tfIdf["TF-IDF"])):
            d['type'] += [tfIdf['type'][index]]
            for i in range(vectorLength):
                if str(i) in tfIdf["TF-IDF"][index]:
                    d[str(i)] += [tfIdf["TF-IDF"][index][str(i)]]
                else:
                    d[str(i)] += [0.0]

        with open(self.VECTOR_FILE_PATH, 'w') as f:
            out = json.dumps(d, indent=4)
            f.write(out)

    def TfIdfTransformation(self, texts):
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        model = TfidfModel(corpus)  # fit model

        preprocesser.saveDictionaryWords(dictionary)
        preprocesser.saveTfIdfCorpus(model, corpus)
        preprocesser.saveDataWithTfIdfInformation(model, corpus)
        preprocesser.saveDictionary(dictionary)
        preprocesser.writeWordVectorsToJson()

preprocesser = FakeNewsPreprocesser(CONFIG_FILE_PATH)

jsonData = preprocesser.readJson()

texts = preprocesser.preprocessText(jsonData)

preprocesser.TfIdfTransformation(texts)

print("end.")