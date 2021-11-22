import numpy as np
import pandas as pd
import logging
import pickle
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def check(self, data):
        val = data[self.column]
        return val>=self.value

class DecisionTreeClassifier:
    def __init__(self, max_depth = 4):
        self.max_depth = max_depth
        self.data = None
        self.tree = None
    def gini(self, rows):
        counts = self.class_counts(rows)
        impurity = 1
        for label in counts:
            probab_of_label = counts[label] / float(len(rows))
            impurity -= probab_of_label**2
        return impurity

    def class_counts(self, rows):
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
    
    def info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def best_split(self,rows):
        best_gain = 0
        best_question = None
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1
        for col in range(n_features):
            values = set([row[col] for row in rows])  
            for val in values:
                question = Question(col, val)
                true_rows, false_rows = self.partition(rows, question)            
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = self.info_gain(true_rows, false_rows, current_uncertainty)            
                if gain > best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question
    def fit(self, features, labels):
        labels = np.expand_dims(labels, axis = 1)
        data = np.append(features, labels, axis = 1)
        self.tree = self.build_tree(data, 0)

    def build_tree(self, data, level):
        gain, question = self.best_split(data)
        if (gain == 0 or level>self.max_depth):
            return Leaf(data)
        true_rows, false_rows = self.partition(data, question)
        true_branch = self.build_tree(true_rows, level+1)
        false_branch = self.build_tree(false_rows, level+1)
        return Node(question, true_branch, false_branch)

    def predict(self, rows):
        ans = []
        for row in rows:
            result = self.classify(row, self.tree)
            ans.append(result)

        return np.array(ans)
    def classify(self, data, node):

        if(isinstance(node, Leaf)):
            return node.pred
        if(node.question.check(data)):
            return self.classify(data, node.true_branch)

        else:
            return self.classify(data, node.false_branch)

    
    def partition(self, rows, qsn):
        true_rows, false_rows = [], []
        for row in rows:
            if qsn.check(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def load(self, filename):
        req_file = open(filename,'rb')
        model = pickle.load(req_file)
        self.tree = model['tree']
        self.max_depth = model['depth']

    def save(self, filename = 'model'):
        try:
            model = {"tree": self.tree, "depth": self.max_depth}
            save_file = open(filename, 'wb')
            pickle.dump(model, save_file)
            save_file.close()

        except:
            logging.exception("There was a problem while saving your model.")

        


class Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
class Leaf:
    def __init__(self, rows):
        self.predictions = self.class_counts(rows)
        self.pred = max(self.predictions, key=self.predictions.get)

    def class_counts(self, rows):
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts