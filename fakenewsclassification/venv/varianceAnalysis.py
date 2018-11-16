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

CONFIG_FILE_PATH = "config.json"

SAMPLE_SIZE = 100
NUMBER_OF_PCA = 3

class FakeNewsVarianceAnalyzer:
    VECTOR_FILE_PATH = None

    def __init__(self, configFilePath):
        with open('config.json') as json_data_file:
            config = json.load(json_data_file)
        #self.VECTOR_FILE_PATH = config["TF_IDF_VECTOR_FILE_PATH"] #[:-5] + "_reduced.json"
        self.VECTOR_FILE_PATH = config["WORD2VEC_VECTOR_FILE_PATH"] #[:-5] + "_reduced.json"

    def readVectors(self):
        return pd.read_json(self.VECTOR_FILE_PATH)

    def getSample(self, vectors, size):
        try:
            chosen_idx = np.random.choice(len(vectors), replace=False, size=size)
            df_trimmed = vectors.iloc[chosen_idx]
            return df_trimmed
        except:
            print("ERROR - Sample size is larger than dataset size! Return with the dataset.")
            return vectors

    def calculatePrincipalComponents(self, vectors, numberOfComponents):
        features = [i for i in vectors.columns.values if i != "type"]

        pca = PCA(n_components=numberOfComponents)
        pca_result = pca.fit_transform(vectors[features].values)

        for i in range(numberOfComponents):
            key = 'pca_' + str(i+1)
            vectors[key] = pca_result[:, i]
        #print('Components: %s' % pcas)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    def showPca(self, vectors):
        chart = ggplot(vectors.loc[:, :], aes(x='pca_1', y='pca_2', color='type')) \
                + geom_point(size=75, alpha=0.8) \
                + ggtitle("First and Second Principal Components")
        chart.show()

    def calculateTsne(self, vectors):
        features = [i for i in vectors.columns.values if i != "type"]

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(vectors.loc[:, features].values)

        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        vectors['x-tsne'] = tsne_results[:, 0]
        vectors['y-tsne'] = tsne_results[:, 1]
        return vectors

    def showTsne(self, vectors):
        chart = ggplot(vectors, aes(x='x-tsne', y='y-tsne', color='type')) \
                + geom_point(size=70, alpha=0.1) \
                + ggtitle("tSNE dimensions")
        chart.show()

    def showPrincipalComponentsAnalysis(self, sample, numberOfPca):
        pca_vectors = sample.loc[:, :].copy()
        self.calculatePrincipalComponents(pca_vectors, numberOfPca)
        self.showPca(pca_vectors)

    def showTsneAnalysis(self, sample):
        tsne_vectors = sample.loc[:, :].copy()
        tsne_vectors = self.calculateTsne(tsne_vectors)
        self.showTsne(tsne_vectors)

analyzer = FakeNewsVarianceAnalyzer(CONFIG_FILE_PATH)
vectors = analyzer.readVectors()
sample = analyzer.getSample(vectors, SAMPLE_SIZE)

analyzer.showPrincipalComponentsAnalysis(sample, NUMBER_OF_PCA)
analyzer.showTsneAnalysis(sample)

print("end.")