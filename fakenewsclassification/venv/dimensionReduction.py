import json
import pandas as pd
from sklearn.decomposition import PCA

class FakeNewsDimensionReduction:
    def __init__(self, configFilePath):
        with open('config.json') as json_data_file:
            config = json.load(json_data_file)
        self.VECTOR_FILE_PATH = config["TF_IDF_VECTOR_FILE_PATH"]

    def readVectors(self):
        return pd.read_json(self.VECTOR_FILE_PATH)

    def pcaDimensionReduction(self, vectors, savedVariance):
        features = [i for i in vectors.columns.values if i != "type"]

        pca = PCA(n_components=savedVariance)
        pca_result = pca.fit_transform(vectors[features].values)
        n = pca_result.shape[1]
        print("Original number of vectors: %i, after reduction: %i" % (len(vectors), n))

        result = dict()
        result["type"] = []
        for i in vectors["type"]:
            result["type"] += [i]
        for i in range(n):
            result[str(i)] = []
        for i in range(n):
            key = str(i)
            for j in range(len(vectors["type"])):
                result[key] += [pca_result[j, i]]
        return result

    def writeReducedVector(self, result, result_file_path):
        with open(result_file_path, 'w') as f:
            out = json.dumps(result, indent=4)
            f.write(out)
