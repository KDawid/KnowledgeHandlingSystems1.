import json
import pandas as pd
from sklearn.utils import shuffle

CONFIG_FILE_PATH = "config.json"

USED_CATEGORY_SET = ['unreliable', 'conspiracy', 'clickbait'] # for 3 mb dataset
NUMBER_OF_INSTANCES_IN_CATEGORIES = 100 # for 3 mb dataset
#USED_CATEGORY_SET = ['conspiracy', 'political', 'fake'] # for 300 mb dataset
#NUMBER_OF_INSTANCES_IN_CATEGORIES = 5000 # for 300 mb dataset

class FakeNewsReader:
    CSV_FILE_PATH = None
    JSON_FILE_PATH = None

    def __init__(self, configFilePath):
        with open(configFilePath) as json_data_file:
            config = json.load(json_data_file)
        self.CSV_FILE_PATH = config["CSV_FILE_PATH"]
        self.JSON_FILE_PATH = config["JSON_FILE_PATH"]

    def readJson(self):
        return pd.read_json(self.JSON_FILE_PATH)

    def convertCsvTojson(self):
        data  = pd.read_csv(self.CSV_FILE_PATH)

        header = data.columns.values
        print("Header: %s" %header)

        out = data.to_json(orient='records')
        with open(self.JSON_FILE_PATH, 'w') as f:
            f.write(out)

    def printCategoryOccurences(self, df):
        categories = df['type']
        catDict = dict()
        for cat in categories:
            if cat not in catDict:
                catDict[cat] = 1
            else:
                catDict[cat] += 1
        import operator
        print("Occurences: %s" % sorted(catDict.items(), key=operator.itemgetter(1), reverse=True))

    def filterCategories(self, jsonData, categorySet, number):
        dataIndexes = []
        for cat in categorySet:
            i = 0
            j = 0
            try:
                while i < number:
                    if jsonData['type'][j] == cat:
                        dataIndexes += [j]
                        i += 1
                    j += 1
            except KeyError:
                print("This database contains only %s instances from category %s" % (i, cat))
        return jsonData.loc[dataIndexes]

    def writeJsonData(self, jsonData):
        jsonData = shuffle(jsonData)
        out = jsonData.to_json(orient='records')
        with open(self.JSON_FILE_PATH, 'w') as f:
            f.write(out)

reader = FakeNewsReader(CONFIG_FILE_PATH)
reader.convertCsvTojson()

jsonData = reader.readJson()
reader.printCategoryOccurences(jsonData)

jsonData = reader.filterCategories(jsonData, USED_CATEGORY_SET, NUMBER_OF_INSTANCES_IN_CATEGORIES)
reader.writeJsonData(jsonData)

print("end.")