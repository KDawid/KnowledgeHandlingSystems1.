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

VECTOR_FILE_PATH = "../../vector.json"

vectors = pd.read_json(VECTOR_FILE_PATH)

print("Vectors before shuffle data:")
print(vectors.head())

vectors = shuffle(vectors)

print("Vectors after shuffle data:")
print(vectors.head())

print("Separate labels from data")
dataset = vectors.drop('type', axis='columns')
dataset_labels = vectors['type']

print("Splitting dataset to training set and validation set")
train, test, train_labels, test_labels = train_test_split(dataset, dataset_labels, test_size=0.20)

print("Decision tree: ")
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, train_labels)
test_pred = decision_tree.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %decision_tree.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(decision_tree, dataset, dataset_labels, cv=10)
print("K-fold score: %s" %np.average(scores))

print("Linear SVM: ")
svm = LinearSVC()
svm.fit(train, train_labels)
test_pred = svm.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %svm.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(svm, dataset, dataset_labels, cv=10)
print("K-fold score: %s" %np.average(scores))

print("Naive bayes: ")
mnb = MultinomialNB()
mnb.fit(train, train_labels)
test_pred = mnb.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %mnb.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(mnb, dataset, dataset_labels, cv=10)
print("K-fold score: %s" %np.average(scores))

print("KNNeighbors: ")
knn = KNeighborsClassifier()
knn.fit(train, train_labels)
test_pred = mnb.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %knn.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(knn, dataset, dataset_labels, cv=10)
print("K-fold score: %s" %np.average(scores))

print("ADA boost: ")
ada = AdaBoostClassifier()
ada.fit(train, train_labels)
test_pred = ada.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %ada.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(ada, dataset, dataset_labels, cv=2)
print("K-fold score: %s" %np.average(scores))

print("Random forest: ")
rf = AdaBoostClassifier()
rf.fit(train, train_labels)
test_pred = rf.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %rf.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(rf, dataset, dataset_labels, cv=2)
print("K-fold score: %s" %np.average(scores))

print("Gradient boost: ")
gbc = GradientBoostingClassifier()
gbc.fit(train, train_labels)
test_pred = gbc.predict(test)

print(confusion_matrix(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print("Validation score: %s" %gbc.score(test, test_labels) )
scores = sklearn.model_selection.cross_val_score(gbc, dataset, dataset_labels, cv=2)
print("K-fold score: %s" %np.average(scores))