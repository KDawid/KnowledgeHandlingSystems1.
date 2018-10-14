import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

JSON_FILE_PATH = "D:\\result.json"
DICTIONARY_FILE_PATH = "D:\\dictionary.csv"

VECTOR_FILE_PATH = "D:\\vector.json"

def getNumberOfVectors(filePath):
    dictionary = pd.read_csv(DICTIONARY_FILE_PATH)
    return len(dictionary)

def writeWordVectorsToJson(filePath):
    jsonData = pd.read_json(JSON_FILE_PATH)
    vectorLength = getNumberOfVectors(DICTIONARY_FILE_PATH)

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
        d['type'] += [tfIdf['type'][0]]
        for i in range(vectorLength):
            if str(i) in tfIdf["TF-IDF"][index]:
                d[str(i)] += [tfIdf["TF-IDF"][index][str(i)]]
            else:
                d[str(i)] += [0.0]

    with open(VECTOR_FILE_PATH, 'w') as f:
        out = json.dumps(d, indent=4)
        f.write(out)

writeWordVectorsToJson(VECTOR_FILE_PATH)

print("end.")