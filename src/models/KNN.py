import numpy as np
class KNN_Model():
    # Constructor
    # Define the number of neighbors to use in prediction
    # Default neighbors = 5
    def __init__(self, k=3, dist=1, weights="uniform"):
        self.k = k
        self.dist = dist
        self.weights = weights
    
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
    def __predict(self, X):
        if (self.dist == 1):
            distances = [self.__euclidean_distance(X, train) for train in self.X_train]
        else:
            # other distance e.g. : Manhattan
            distances = [self.__manhattan_distance(X, train) for train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = None
        if(self.weights == "uniform"):
            most_common = np.bincount(k_nearest_labels)
        elif(self.weights == "distance"):
            weights = 1 / (np.array(distances) + 1e-10)
            chosen_weights = [weights[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels, weights=chosen_weights)
        return np.argmax(most_common)

    
    def predict(self, X):
        X_ = X.to_numpy()
        result = [self.__predict(test) for test in X_]
        return np.array(result)
    
    # Distance Method for KNN
    def __euclidean_distance(self,p1,p2):
        p1_numeric = np.array(p1, dtype=float)
        p2_numeric = np.array(p2, dtype=float)
        return np.sqrt(np.sum((p1_numeric - p2_numeric) ** 2))
    
    def __manhattan_distance(self, p1, p2):
        p1_numeric = np.array(p1, dtype=float)
        p2_numeric = np.array(p2, dtype=float)
        return np.abs(p1_numeric - p2_numeric)