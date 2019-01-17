from sklearn.externals import joblib
import os

class NewsClassifier:

    def __init__(self, model_file=None):

        self.trained = False

        if model_file is not None:
            self.load_from_file(model_file)

        self.stances = ['L', 'R', 'C']

    def load_from_file(self, model_file):
        if not os.path.isfile(model_file):
            raise Exception("Model file does not exist")

        print("Loading classifier...")
        self.clf = joblib.load(model_file)
        print("Classifier loaded")
        self.trained = True


    def predict(self, text):
        if not self.trained:
            raise Exception("Model is not trained")

        stance = self.stances[self.clf.predict([text])[0]]
        return stance