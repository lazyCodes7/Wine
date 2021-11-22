import numpy as np
import pandas as pd
class NBClassifier:
    def __init__(self):
        self.data = None
        self.summarized = None

    def fit(self, features, labels):
        labels = np.expand_dims(labels, axis = 1)
        data = np.append(features, labels, axis = 1)
        self.data = data
        self.summarized = self.summarize_by_class()

    def seperate_by_class(self):
        seperator = {}
        for vectors in self.data:
            class_value = vectors[-1]
            if class_value not in seperator:
                seperator[class_value] = []

            else:
                seperator[class_value].append(vectors)
        return seperator

    def summarize_dataset(self, rows):
        summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*rows)]
        del(summaries[-1])
        return summaries

    def summarize_by_class(self):
        separated = self.seperate_by_class()
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(self, x, mean, stdev):
        exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, row):
        total_rows = sum([self.summarized[label][0][2] for label in self.summarized])
        probabilities = dict()
        for class_value, class_summaries in self.summarized.items():
            probabilities[class_value] = self.summarized[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    def evaluate(self,row):
        probabilities = self.calculate_class_probabilities(row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
    def predict(self, test_set):
        preds = []
        for rows in test_set:
            output = self.evaluate(rows)
            preds.append(output)
        return np.array(preds)



