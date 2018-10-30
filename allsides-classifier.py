from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import csv
import sys

file_path = "./data/allsides.csv"
n_grams = 1
train_pct = 0.8


def main():
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

    train_split = int(train_pct*(len(docs)))
    train_data = docs[:train_split]
    train_labels = labels[:train_split]
    test_data = docs[train_split:]
    test_labels = labels[train_split:]

    count_vect = CountVectorizer(ngram_range=(1,n_grams))
    tfidf_transformer = TfidfTransformer()    
    clf = LinearSVC()

    train_counts = count_vect.fit_transform(train_data)
    train_tfidf = tfidf_transformer.fit_transform(train_counts)

    clf.fit(train_tfidf, train_labels)

    test_counts = count_vect.transform(test_data)
    test_tfidf = tfidf_transformer.transform(test_counts)

    predicted = clf.predict(test_tfidf)
    

    # Evaluate accuracy of model
    acc = accuracy_score(test_labels, predicted)
    print("Accuracy Score: {}".format(acc))

if __name__ == "__main__":
    main()