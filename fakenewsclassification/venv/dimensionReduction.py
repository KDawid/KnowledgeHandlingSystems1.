import json
import pandas as pd
from sklearn.decomposition import PCA

CONFIG_FILE_PATH = "config.json"

NUMBER_OF_ATTRIBUTES = 300

class FakeNewsDimensionReduction:
    def __init__(self, configFilePath):
        with open('config.json') as json_data_file:
            config = json.load(json_data_file)
        self.VECTOR_FILE_PATH = config["TF_IDF_VECTOR_FILE_PATH"]
        #self.VECTOR_FILE_PATH = config["WORD2VEC_VECTOR_FILE_PATH"]

    def readVectors(self):
        return pd.read_json(self.VECTOR_FILE_PATH)

    def pcaDimensionReduction(self, vectors, numberOfComponents):
        features = [i for i in vectors.columns.values if i != "type"]

        pca = PCA(n_components=numberOfComponents)
        pca_result = pca.fit_transform(vectors[features].values)

        result = dict()
        result["type"] = []
        for i in vectors["type"]:
            result["type"] += [i]
        for i in range(numberOfComponents):
            result[str(i)] = []
        for i in range(numberOfComponents):
            key = str(i)
            for j in range(len(vectors["type"])):
                result[key] += [pca_result[j, i]]
        return result

    def writeReducedVector(self, result, result_file_path):
        with open(result_file_path, 'w') as f:
            out = json.dumps(result, indent=4)
            f.write(out)

reducer = FakeNewsDimensionReduction(CONFIG_FILE_PATH)
vectors = reducer.readVectors()
result_file_path = reducer.VECTOR_FILE_PATH[:-5] + "_reduced.json"

result = reducer.pcaDimensionReduction(vectors, NUMBER_OF_ATTRIBUTES)

reducer.writeReducedVector(result, result_file_path)

print(result_file_path)

print("end.")