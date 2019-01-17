from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.externals import joblib
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import csv
import sys

file_path = "./data/allsides.csv"
model_file_path = "./models/article-classifier_8000x3.pkl"
save = False
n_grams = 1
train_pct = 0.8

def load_data():
    csv.field_size_limit(sys.maxsize)
    docs = []
    labels = []

    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row[2] == "l":
                docs.append(row[1])
                labels.append(0)
            elif row[2] == "r":
                docs.append(row[1])
                labels.append(1)
            elif row[2] == "c":
                docs.append(row[1])
                labels.append(2)
    
    docs = np.array(docs)
    labels = np.array(labels)

    return docs, labels

def main():
    docs, labels = load_data()

    train_split = int(train_pct*(len(docs)))
    train_data = docs[:train_split]
    train_labels = labels[:train_split]
    test_data = docs[train_split:]
    test_labels = labels[train_split:]

    clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    #('feature_selection', SelectKBest(chi2, k=50000)),
                    ('clf', MLPClassifier(verbose=True, max_iter=8))])
    print(train_data.shape)
    
    # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    #                 'tfidf__use_idf': (True, False)
    # }
    # print("Begin grid search...")
    # gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, scoring="accuracy", verbose=10)
    # print("Training model...")
    # gs_clf = gs_clf.fit(train_data, train_labels)

    # print("Best score: " + gs_clf.best_score_)
    # print(gs_clf.best_params_)
    
    clf.fit(train_data, train_labels)
    #clf = joblib.load(model_file_path)

    predicted = clf.predict(test_data)

    # Evaluate the model
    acc = accuracy_score(test_labels, predicted)
    print("Accuracy Score: {}".format(acc))
    report = classification_report(test_labels, predicted)
    print(report)

    if save:
        joblib.dump(clf, model_file_path)


if __name__ == "__main__":
    main()