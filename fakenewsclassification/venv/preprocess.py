import pandas as pd
from gensim.models import TfidfModel
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string
from pprint import pprint  # pretty-printer

CSV_FILE_PATH = "D:\\data3mb.csv"
JSON_FILE_PATH = "D:\\data.json"
DICTIONARY_FILE_PATH = "D:\\dictionary.dict"

STOPLIST = set('for a of the and to in'.split())
MIN_FREQUENCY = 5

def convertCsvTojson(csvFilePath, jsonFilePath):
    data  = pd.read_csv(csvFilePath)

    header = data.columns.values
    print("Header: %s" %header)

    out = data.to_json(orient='records')
    with open(jsonFilePath, 'w') as f:
        f.write(out)

def applyMinFrequency(texts, i):
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
          frequency[token] += 1

    return [[token for token in text if frequency[token] > i]
             for text in texts]

def printDictionaryItems(dictionary):
    for (id, word) in dictionary.items():
        print(word)

#convertCsvTojson(CSV_FILE_PATH, JSON_FILE_PATH)
jsonData = pd.read_json(JSON_FILE_PATH)

#texts = [[word for word in document.lower().split() if word not in STOPLIST] for document in jsonData['content']]
#texts = applyMinFrequency(texts, MIN_FREQUENCY)

texts = [text.replace("“", "").replace("…", "").replace("‘", "").replace("”", "").replace("’", "").replace("—", "") for text in jsonData['content']]
texts = [preprocess_string(text) for text in texts]

dictionary = corpora.Dictionary(texts)
#dictionary.save(DICTIONARY_FILE_PATH)
corpus = [dictionary.doc2bow(text) for text in texts]
model = TfidfModel(corpus)  # fit model
vector = model[corpus[0]]

#printDictionaryItems(dictionary)

#for c in range(0,len(corpus)) :
#    print("%s. vector: %s" %(c, model[corpus[c]]))

print("end.")