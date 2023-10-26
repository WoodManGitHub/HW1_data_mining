import numpy as np
from sklearn.preprocessing import LabelEncoder
from graphviz import Digraph

class C45DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = np.unique(y_encoded)
        self.tree = self._build_tree(X, y_encoded, feature_names)
        
    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def _information_gain(self, X, y, feature):
        total_entropy = self._entropy(y)
        unique_values = np.unique(X[:, feature])
        weighted_entropy = 0

        for value in unique_values:
            subset_indices = np.where(X[:, feature] == value)[0]
            subset_entropy = self._entropy(y[subset_indices])
            weighted_entropy += (len(subset_indices) / len(y)) * subset_entropy

        information_gain = total_entropy - weighted_entropy
        return information_gain

    def _choose_best_feature(self, X, y, features):
        information_gains = [self._information_gain(X, y, feature) for feature in features]
        best_feature_index = np.argmax(information_gains)
        return features[best_feature_index]

    def _build_tree(self, X, y, feature_names, depth=0):
        if len(np.unique(y)) == 1:
            return y[0]
        if len(X) == 0 or depth >= 5:  # Stopping criteria
            return np.argmax(np.bincount(y))

        features = list(range(X.shape[1]))
        best_feature = self._choose_best_feature(X, y, features)
        tree = {feature_names[best_feature]: {}}

        unique_values = np.unique(X[:, best_feature])
        for value in unique_values:
            value_indices = np.where(X[:, best_feature] == value)[0]
            sub_X = X[value_indices]
            sub_y = y[value_indices]
            sub_features = [f for f in features if f != best_feature]
            tree[feature_names[best_feature]][value] = self._build_tree(sub_X, sub_y, feature_names, depth + 1)

        return tree

    def predict(self, X):
        y_encoded = [self._predict_tree(self.tree, x) for x in X]
        return self.label_encoder.inverse_transform(y_encoded)

    def _predict_tree(self, tree, x):
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        value = x[self.feature_names.index(feature)]
        if value not in tree[feature]:
            return np.argmax(np.bincount(list(self.classes)))
        return self._predict_tree(tree[feature][value], x)

    def visualize_tree(self, dot=None, parent=None, parent_branch_label=None):
        if dot is None:
            dot = Digraph(comment='Decision Tree')
        
        if not isinstance(self.tree, dict):
            dot.node(str(self.tree), label=str(self.tree))
            if parent is not None:
                dot.edge(str(parent), str(self.tree), label=parent_branch_label)
        else:
            feature = list(self.tree.keys())[0]
            dot.node(str(feature), label=self.feature_names[feature])
            if parent is not None:
                dot.edge(str(parent), str(feature), label=parent_branch_label)

            for value in self.tree[feature]:
                sub_tree = self.tree[feature][value]
                self.__class__(tree=sub_tree, feature_names=self.feature_names).visualize_tree(dot, parent=feature, parent_branch_label=str(value))

        return dot

