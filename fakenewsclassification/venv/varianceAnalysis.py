#Source: https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
from ggplot import *
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

VECTOR_FILE_PATH = config["VECTOR_FILE_PATH"]

SAMPLE_SIZE = 100
NUMBER_OF_PCA = 3

def readVectors(filePath):
    return pd.read_json(filePath)

def getSample(vectors, size):
    try:
        chosen_idx = np.random.choice(len(vectors), replace=False, size=size)
        df_trimmed = vectors.iloc[chosen_idx]
        return df_trimmed
    except:
        print("ERROR - Sample size is larger than dataset size! Return with the dataset.")
        return vectors

def calculatePrincipalComponents(vectors, numberOfComponents):
    features = [i for i in vectors.columns.values if i != "type"]

    pca = PCA(n_components=numberOfComponents)
    pca_result = pca.fit_transform(vectors[features].values)

    for i in range(numberOfComponents):
        key = 'pca_' + str(i+1)
        vectors[key] = pca_result[:, i]
    #print('Components: %s' % pcas)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

def showPca(vectors):
    chart = ggplot(vectors.loc[:, :], aes(x='pca_1', y='pca_2', color='type')) \
            + geom_point(size=75, alpha=0.8) \
            + ggtitle("First and Second Principal Components")
    chart.show()

def calculateTsne(vectors):
    features = [i for i in vectors.columns.values if i != "type"]

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(vectors.loc[:, features].values)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    tsne_vectors['x-tsne'] = tsne_results[:, 0]
    tsne_vectors['y-tsne'] = tsne_results[:, 1]

def showTsne(vectors):
    chart = ggplot(vectors, aes(x='x-tsne', y='y-tsne', color='type')) \
            + geom_point(size=70, alpha=0.1) \
            + ggtitle("tSNE dimensions")
    chart.show()

vectors = readVectors(VECTOR_FILE_PATH)
sample = getSample(vectors, SAMPLE_SIZE)

pca_vectors = sample.loc[:,:].copy()
calculatePrincipalComponents(pca_vectors, NUMBER_OF_PCA)
showPca(pca_vectors)

tsne_vectors = sample.loc[:,:].copy()
calculateTsne(tsne_vectors)
showTsne(tsne_vectors)

print("end.")