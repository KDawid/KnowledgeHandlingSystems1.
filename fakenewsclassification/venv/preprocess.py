import pandas as pd
import json
from gensim.models import TfidfModel
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string
from pprint import pprint  # pretty-printer
from unidecode import unidecode

CSV_FILE_PATH = "D:\\data3mb.csv"
JSON_FILE_PATH = "D:\\data.json"
DICTIONARY_FILE_PATH = "D:\\dictionary.csv"
SAVE_WORDS_FILE_PATH = "D:\\words.txt"
CORPUS_FILE_PATH = "D:\\corpus.txt"
RESULT_FILE_PATH = "D:\\result.json"

STOPLIST = set('for a of the and to in'.split())
MIN_FREQUENCY = 5

def convertCsvTojson(csvFilePath, jsonFilePath):
    data  = pd.read_csv(csvFilePath)

    header = data.columns.values
    print("Header: %s" %header)

    out = data.to_json(orient='records')
    with open(jsonFilePath, 'w') as f:
        f.write(out)

def implementedPreprocessing(data):
    texts = [[word for word in document.lower().split() if word not in STOPLIST] for document in data]
    texts = applyMinFrequency(texts, MIN_FREQUENCY)
    return text

def applyMinFrequency(texts, i):
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
          frequency[token] += 1
    return [[token for token in text if frequency[token] > i] for text in texts]

def printDictionaryItems(dictionary):
    for (id, word) in dictionary.items():
        print(id, word)

def saveDictionaryWords(dictionary, filePath):
    wordSet = set()
    for (id, word) in dictionary.items():
        wordSet.add(word)
    with open(filePath, 'w') as f:
        for word in sorted(wordSet):
            f.write(unidecode(word) + "\n")

def saveDictionary(dictionary, filePath):
    with open(filePath, 'w') as f:
        for (id, word) in dictionary.items():
            f.write(str(id) + "," + unidecode(word) + "\n")

def saveTfIdfCorpus(model, filePath):
    with open(filePath, 'w') as f:
        f.write("[")
        for i in range(len(corpus)):
            f.write(json.dumps(model[corpus[i]]))
            if i != len(corpus) - 1:
                f.write(",")
        f.write("]")

def saveDataWithTfIdfInformation(model, dataFilePath, resultFilePath):
    tfCorpus = [model[corpus[i]] for i in range(len(corpus))]
    result = json.load(open(dataFilePath))
    for i in range(len(tfCorpus)):
        result[i]["TF-IDF"] = dict(tfCorpus[i])
    with open(resultFilePath, 'w') as f:
        out = json.dumps(result, indent=4)
        f.write(out)

convertCsvTojson(CSV_FILE_PATH, JSON_FILE_PATH)
jsonData = pd.read_json(JSON_FILE_PATH)

texts = jsonData['content']
texts = [text.replace("“", " ").replace("…", " ").replace("‘", " ").replace("”", " ").replace("’", " ").replace("—", " ").replace("-", " ").replace("  ", " ") for text in jsonData['content']]
texts = [preprocess_string(text) for text in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
model = TfidfModel(corpus)  # fit model

saveDictionaryWords(dictionary, SAVE_WORDS_FILE_PATH)
saveTfIdfCorpus(model, CORPUS_FILE_PATH)
saveDataWithTfIdfInformation(model, JSON_FILE_PATH, RESULT_FILE_PATH)
saveDictionary(dictionary,DICTIONARY_FILE_PATH)

print("end.")