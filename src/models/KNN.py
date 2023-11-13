import numpy as np

class KNN_Model():
    # Constructor
    # Define the number of neighbors to use in prediction
    # Default neighbors = 5
    def __init__(self, k=3, dist=1):
        self.k = k
        self.dist = dist
    
    # Fit Method
    # Fit the training dataset to the model
    # X are the features, Y is the target
    def fit(self, X, y):
        self.features = X.columns
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()
    
    # Predict Method
    # Prediction using KNN Algorithm
    # X is the data to be predicted
    # Return Predicted Data
    def predict(self, X):
        X_ = X.to_numpy()
        result = np.empty(X_.shape[0])
        for i, train in enumerate(X_):
            distances = [ self.__euclidean_distance(train, data) for data in self.X_train ]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels)
            result[i] = np.argmax(most_common)
        return result

    # Distance Method for KNN
    def __euclidean_distance(self,p1,p2):
        p1_numeric = np.array(p1, dtype=float)
        p2_numeric = np.array(p2, dtype=float)
        return np.sqrt(np.sum((p1_numeric - p2_numeric) ** 2))
    
    def __manhattan_distance(self, p1, p2):
        p1_numeric = np.array(p1, dtype=float)
        p2_numeric = np.array(p2, dtype=float)
        return np.abs(p1_numeric - p2_numeric)