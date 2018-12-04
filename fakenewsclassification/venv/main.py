import json
from reader import FakeNewsReader
from preprocess import FakeNewsPreprocesser
from preprocess import PREPROCESS_TYPE
from preprocess import TRANSFORMATION_TYPE
from dimensionReduction import FakeNewsDimensionReduction
from varianceAnalysis import FakeNewsVarianceAnalyzer
from classify import FakeNewsClassifier
from classify import Evaluation

CONFIG_FILE_PATH = "config.json"
VECTOR_FILE_PATHS = dict()

#for reader
USED_CATEGORY_SET = ['unreliable', 'conspiracy', 'clickbait'] # for 3 mb dataset
NUMBER_OF_INSTANCES_IN_CATEGORIES = 100 # for 3 mb dataset
#USED_CATEGORY_SET = ['conspiracy', 'political', 'fake' ] # for 300 mb dataset
#NUMBER_OF_INSTANCES_IN_CATEGORIES = 5000 # for 300 mb dataset

#for preprocess
# more preprocess options: https://radimrehurek.com/gensim/parsing/preprocessing.html

#for dimension reduction
SAVED_VARIANCE = 0.9

#for varianceAnalysis
SAMPLE_SIZE = 100
NUMBER_OF_PCA = 3

def readData():
    reader = FakeNewsReader(CONFIG_FILE_PATH)
    reader.convertCsvTojson()

    jsonData = reader.readJson()
    reader.printCategoryOccurences(jsonData)

    jsonData = reader.filterCategories(jsonData, USED_CATEGORY_SET, NUMBER_OF_INSTANCES_IN_CATEGORIES)
    reader.writeJsonData(jsonData)

def preprocessData(preprocessType, transformationType, filePath):
    preprocesser = FakeNewsPreprocesser(CONFIG_FILE_PATH)
    jsonData = preprocesser.readJson()
    texts = preprocesser.preprocessText(jsonData, preprocessType)
    preprocesser.transformData(texts, jsonData, transformationType, filePath)

def reduceDimensions(vectorType):
    reducer = FakeNewsDimensionReduction(CONFIG_FILE_PATH)
    reducer.VECTOR_FILE_PATH = vectorType
    vectors = reducer.readVectors()
    result_file_path = vectorType[:-5] + "_reduced.json"

    result = reducer.pcaDimensionReduction(vectors, SAVED_VARIANCE)
    reducer.writeReducedVector(result, result_file_path)

def varianceAnalysis(vectorFilePath):
    analyzer = FakeNewsVarianceAnalyzer(CONFIG_FILE_PATH)
    analyzer.VECTOR_FILE_PATH = vectorFilePath
    vectors = analyzer.readVectors()
    sample = analyzer.getSample(vectors, SAMPLE_SIZE)

    analyzer.savePrincipalComponentsAnalysis(sample, NUMBER_OF_PCA, vectorFilePath)
    analyzer.saveTsneAnalysis(sample, vectorFilePath)

def classify(vectorFilePath):
    classifier = FakeNewsClassifier(CONFIG_FILE_PATH, vectorFilePath)
    result = classifier.findBestClassifier(Evaluation.BOTH)
    with open(vectorFilePath[:-5] + "_result.json", 'w') as f:
        out = json.dumps(result, indent=4)
        f.write(out)

def updateFilePaths():
    for i in list(VECTOR_FILE_PATHS):
        VECTOR_FILE_PATHS[i + "_red"] = VECTOR_FILE_PATHS[i][:-5] + "_reduced.json"

with open('config.json') as json_data_file:
    config = json.load(json_data_file)
    for prep in [type.value for type in PREPROCESS_TYPE]:
        VECTOR_FILE_PATHS['tf-idf_' + prep] = config["TF_IDF_VECTOR_FILE_PATH"][:-5] + "_" + prep + ".json"
        VECTOR_FILE_PATHS['word2vec_' + prep] = config["WORD2VEC_VECTOR_FILE_PATH"][:-5] + "_" + prep + ".json"
        VECTOR_FILE_PATHS['w2v-tfidf_' + prep] = config["WORD2VEC_TFIDF_VECTOR_FILE_PATH"][:-5] + "_" + prep + ".json"

print("")
print(VECTOR_FILE_PATHS)
print("")

#READER
print("Reading data")
readData()
print("Reading data was successful.\n")

#PREPROCESSER
for preprocessType in PREPROCESS_TYPE:
    for transformationType in TRANSFORMATION_TYPE:
        print("Preprocess: %s, transform: %s" %(preprocessType.value, transformationType.value))
        preprocessData(preprocessType, transformationType, VECTOR_FILE_PATHS[transformationType.value + "_" + preprocessType.value])
        print("Preprocess is done.\n")

#DIMENSION REDUCTION
for vectorType in VECTOR_FILE_PATHS.values():
    print("Dimension reduction on %s" % vectorType)
    reduceDimensions(vectorType)
    print("Dimension reduction is done.\n")
updateFilePaths()

#VARIANCE ANALYSIS
for vectorType in VECTOR_FILE_PATHS.values():
    print("Variance analysis on %s" % vectorType)
    varianceAnalysis(vectorType)
    print("Variance analysis is done.\n")

#CLASSIFICATION
for vectorType in VECTOR_FILE_PATHS.values():
    print("Classification on %s" % vectorType)
    classify(vectorType)
    print("Classification is done.\n")

print("end.")