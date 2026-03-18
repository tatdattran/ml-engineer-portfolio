import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load CSV file"""
    return pd.read_csv(path)

def split_features_target(df, target='SalePrice'):
    """Split dataframe into X and y"""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
