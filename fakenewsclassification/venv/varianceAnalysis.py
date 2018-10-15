import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

JSON_FILE_PATH = "D:\\result.json"
DICTIONARY_FILE_PATH = "D:\\dictionary.csv"

VECTOR_FILE_PATH = "D:\\vector.json"

USED_CATEGORY_SET = ['unreliable', 'conspiracy', 'clickbait'] # for 3 mb dataset

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

def readVectors(filePath):
    return pd.read_json(filePath)

#writeWordVectorsToJson(VECTOR_FILE_PATH)
vectors = readVectors(VECTOR_FILE_PATH)

print(vectors)
features = [i for i in vectors.columns.values if i != "type"]
# Separating out the features
x = vectors.loc[:, features].values
# Separating out the target
y = vectors.loc[:,['type']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, vectors[['type']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = USED_CATEGORY_SET
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show(block=True)

print("end.")