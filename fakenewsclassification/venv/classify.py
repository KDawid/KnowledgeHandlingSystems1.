import json
from enum import Enum
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

CONFIG_FILE_PATH = "config.json"

TEST_SIZE = 0.20
CROSS_VALIDATION = 10

class Evaluation(Enum):
    CROSS_VALIDATION = 0
    VALIDATION_SET = 1
    BOTH = 2

class Classifiers(Enum):
    #KEY = [Classifier function, classifier name]
    DECISION_TREE = [DecisionTreeClassifier, "Decision tree"]
    SVM = [LinearSVC, "Linear SVM"]
    #NAIVE_BAYES = [MultinomialNB, "Naive Bayes"]
    KNN = [KNeighborsClassifier, "K-Nearest Neighbors"]
    ADA_BOOST = [AdaBoostClassifier, "ADA boost"]
    RANDOM_FOREST = [AdaBoostClassifier, "Random forest"]
    GRADIENT_BOOST = [GradientBoostingClassifier, "Gradient boost"]

class FakeNewsClassifier:
    VECTOR_FILE_PATH = None

    dataset = None
    dataset_labels = None
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None

    classifierType = None
    classifierName = None

    def __init__(self, configFilePath):
        with open('config.json') as json_data_file:
            config = json.load(json_data_file)
        #self.VECTOR_FILE_PATH = config["TF_IDF_VECTOR_FILE_PATH"] #[:-5] + "_reduced.json"
        self.VECTOR_FILE_PATH = config["WORD2VEC_VECTOR_FILE_PATH"] #[:-5] + "_reduced.json"
        self.readData()

    def readData(self):
        vectors = pd.read_json(self.VECTOR_FILE_PATH)
        self.dataset = vectors.drop('type', axis='columns')
        self.dataset_labels = vectors['type']
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.dataset, self.dataset_labels, test_size=TEST_SIZE)

    def setClassifierType(self, type):
        self.classifierType = type.value[0]
        self.classifierName = type.value[1]

    def buildModel(self, train_data, train_labels):
        print("%s: " % self.classifierName)
        model = self.classifierType()
        model.fit(train_data, train_labels)
        return model

    def evaluateOnValidationData(self, model):
        test_pred = model.predict(self.test_data)
        print(confusion_matrix(self.test_labels, test_pred))
        print(classification_report(self.test_labels, test_pred))
        score = model.score(self.test_data, self.test_labels)
        print("Validation score: %s" % score)
        return score

    def crossValidate(self, model):
        scores = sklearn.model_selection.cross_val_score(model, self.dataset, self.dataset_labels,
                                                         cv=CROSS_VALIDATION)
        score = np.average(scores)
        print("K-fold score: %s" % score)
        return score

    def makeClassification(self, eval):
        result = dict()
        if eval != Evaluation.CROSS_VALIDATION :
            #Evaluate on validation data
            model = self.buildModel(self.train_data, self.train_labels)
            result["test"] = self.evaluateOnValidationData(model)

        if eval != Evaluation.VALIDATION_SET:
            #Evaluate with cross validation
            model = self.buildModel(self.dataset, self.dataset_labels)
            result["cross"] = self.crossValidate(model)
        return result

    def classify(self, type, eval=Evaluation.BOTH):
        self.setClassifierType(type)
        return self.makeClassification(eval)

    def findBestClassifier(self, eval=Evaluation.BOTH):
        result = dict()
        for i in Classifiers:
            result[i.value[1]] = classifier.classify(i, eval)
            print("-----------------------------------------------------------------")
        if "test" in result[next(iter(result))]:
            print("Best accuracy using validation set: %s, classifier: %s" % max(
                [(value["test"], key) for key, value in result.items()]))
        if "cross" in result[next(iter(result))]:
            print("Best accuracy using cross-validation: %s, classifier: %s" % max(
                [(value["cross"], key) for key, value in result.items()]))

classifier = FakeNewsClassifier(CONFIG_FILE_PATH)

#classifier.classify(Classifiers.SVM, Evaluation.CROSS_VALIDATION)
classifier.findBestClassifier(Evaluation.BOTH)

print("end.")