import json
import pandas as pd

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

CSV_FILE_PATH = config["CSV_FILE_PATH"]
JSON_FILE_PATH = config["JSON_FILE_PATH"]

USED_CATEGORY_SET = ['unreliable', 'conspiracy', 'clickbait'] # for 3 mb dataset
NUMBER_OF_INSTANCES_IN_CATEGORIES = 100 # for 3 mb dataset
#USED_CATEGORY_SET = ['conspiracy', 'political', 'fake'] # for 300 mb dataset
#NUMBER_OF_INSTANCES_IN_CATEGORIES = 5000 # for 300 mb dataset

def convertCsvTojson(csvFilePath, jsonFilePath):
    data  = pd.read_csv(csvFilePath)

    header = data.columns.values
    print("Header: %s" %header)

    out = data.to_json(orient='records')
    with open(jsonFilePath, 'w') as f:
        f.write(out)

def printCategoryOccurences(df):
    categories = df['type']
    catDict = dict()
    for cat in categories:
        if cat not in catDict:
            catDict[cat] = 1
        else:
            catDict[cat] += 1
    import operator
    print("Occurences: %s" % sorted(catDict.items(), key=operator.itemgetter(1), reverse=True))

def filterCategories(jsonData, categorySet, number):
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

def writeJsonData(jsonData, filePath):
    out = jsonData.to_json(orient='records')
    with open(JSON_FILE_PATH, 'w') as f:
        f.write(out)

convertCsvTojson(CSV_FILE_PATH, JSON_FILE_PATH)
jsonData = pd.read_json(JSON_FILE_PATH)

printCategoryOccurences(jsonData)
jsonData = filterCategories(jsonData, USED_CATEGORY_SET, NUMBER_OF_INSTANCES_IN_CATEGORIES)
writeJsonData(jsonData, JSON_FILE_PATH)

print("end.")