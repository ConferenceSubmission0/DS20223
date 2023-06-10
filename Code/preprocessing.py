import pandas as pd
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # Calculate the mean along each column of X
        self.mean = np.mean(X, axis=0)
        # Calculate the standard deviation along each column of X
        self.std = np.std(X, axis=0)

    def transform(self, X):
        # Scale the input data X using the mean and standard deviation
        X_scaled = (X - self.mean) / self.std
        return X_scaled

    def fit_transform(self, X):
        # Fit the scaler to the input data X and transform X
        self.fit(X)
        X_scaled = self.transform(X)
        return X_scaled

def one_hot_encoder(data, cols, target, classf=0):
    if classf == 0:
        # Perform one-hot encoding on the specified columns in the data
        one_hot_encoded_data = pd.get_dummies(data, columns=cols)
        return one_hot_encoded_data
    else:
        # Extract the target column from the data
        data_target = pd.DataFrame(data[target], columns=[target])
        # Extract the feature columns from the data
        data_features = pd.DataFrame(data.drop(columns=[target]))
        # Perform one-hot encoding on the feature columns
        one_hot_encoded_data = pd.get_dummies(data_features, columns=cols)
        # Concatenate the one-hot encoded features with the target column
        one_hot_encoded_data = pd.concat([one_hot_encoded_data, data_target], axis=1)
        return one_hot_encoded_data

def standardization(data, target, classf=0):
    # Create an instance of the StandardScaler class
    scale = StandardScaler()
    # Get the column names of the data
    cols = data.columns

    if classf == 0:
        # Perform standardization on the entire data
        scaled = scale.fit_transform(data)
        data = pd.DataFrame(scaled, columns=cols)
    else:
        # Get the feature columns excluding the target column
        features = data.columns[:-1]
        # Extract the target column from the data
        data_target = pd.DataFrame(data[target], columns=[target])
        # Extract the feature columns from the data
        data_features = pd.DataFrame(data.drop(columns=[target]))
        # Perform standardization on the feature columns
        scaled = pd.DataFrame(scale.fit_transform(data_features), columns=features)
        # Concatenate the scaled features with the target column
        data = pd.concat([scaled, data_target], axis=1)
    return data

def preprocessing(data, target, classf=0):
    # Remove rows with missing values from the data
    data = data.dropna()

    if classf == 0:
        # Get the column names of the data that are categorical features
        cols = [c for c in data.columns if data[c].dtype == object and c != target]
        # Perform one-hot encoding on the categorical features
        data = one_hot_encoder(data, cols, target, classf)
        # Perform standardization on the data
        data = standardization(data, target, classf)
    else:
        # Get the column names of the data that are categorical features
        cols = [c for c in data.columns if data[c].dtype == object and c != target]
        # Perform one-hot encoding on the categorical features
        data = one_hot_encoder(data, cols, target, classf)
        # Perform standardization on the data
        data = standardization(data, target, classf)
    return data
