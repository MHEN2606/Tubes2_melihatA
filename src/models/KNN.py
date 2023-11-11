class KNN_Model():
    # Constructor
    # Define the number of neighbors to use in prediction
    # Default neighbors = 5
    def __init__(self, k: int = 5):
        self.k = k

    # Fit Method
    # Fit the training dataset to the model
    # X are the features, Y is the target
    def fit(self, X, y):
        self.features= X.columns
        self.target = y.to_numpy()
        self.data = X.to_numpy()

    # Predict Method
    # Prediction using KNN Algorithm
    # X is the data to be predicted
    # Return Predicted Data
    def predict(self, X):
        return self