class NaiveBayes_Model():
    # Constructor
    def __init__(self):
        pass

    # Fit Method
    # Fit the training dataset to the model
    # X are the features, Y is the target
    def fit(self, X, y):
        self.features= X.columns
        self.target = y.to_numpy()
        self.data = X.to_numpy()

    # Predict Method
    # Prediction using Naive Bayes Algorithm
    # X is the data to be predicted
    # Return predicted data
    def predict(self, X):
        return self