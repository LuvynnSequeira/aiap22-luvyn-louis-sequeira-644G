"""
Data Preprocessing Module for Phishing Website Detection

Handles data loading, cleaning, encoding, and train-test split.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os


class DataPreprocessor:
    """Handles data preprocessing for the phishing detection pipeline."""
    
    def __init__(self, db_path='data/phishing.db', test_size=0.2, random_state=42):
        """Initialize preprocessor with database path and split parameters."""
        self.db_path = db_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def load_data(self):
        """Load data from SQLite database."""
        print("Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM phishing_data", conn)
        conn.close()
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def preprocess_features(self, df):
        """Preprocess features: handle missing values and encode categoricals."""
        print("\nPreprocessing features...")
        df = df.copy()
        
        # Drop the unnamed index column
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Separate features and target
        if 'label' in df.columns:
            y = df['label']
            X = df.drop('label', axis=1)
        else:
            y = None
            X = df
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Handle missing values in numerical features
        if X[numerical_cols].isnull().sum().sum() > 0:
            print(f"Imputing {X[numerical_cols].isnull().sum().sum()} missing values...")
            X[numerical_cols] = self.imputer.fit_transform(X[numerical_cols])
        
        # Encode categorical features using frequency encoding
        for col in categorical_cols:
            print(f"Encoding categorical feature: {col}")
            # Frequency encoding
            freq_encoding = X[col].value_counts().to_dict()
            X[col + '_freq'] = X[col].map(freq_encoding)
            
            # Label encoding for backup
            le = LabelEncoder()
            X[col + '_label'] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
            # Drop original categorical column
            X = X.drop(col, axis=1)
        
        self.feature_names = X.columns.tolist()
        print(f"Total features after preprocessing: {len(self.feature_names)}")
        
        if y is not None:
            return X, y
        return X
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler."""
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y):
        """Split data with stratification to maintain class distribution."""
        print(f"\nSplitting data (test_size={self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training set class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Test set class distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        # Load data
        df = self.load_data()
        
        # Preprocess features
        X, y = self.preprocess_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor objects for later use."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_dict = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_dict, f)
        
        print(f"\nPreprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load saved preprocessor objects."""
        with open(filepath, 'rb') as f:
            preprocessor_dict = pickle.load(f)
        
        self.scaler = preprocessor_dict['scaler']
        self.label_encoders = preprocessor_dict['label_encoders']
        self.imputer = preprocessor_dict['imputer']
        self.feature_names = preprocessor_dict['feature_names']
        
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("="*60)
    print("TESTING DATA PREPROCESSING MODULE")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()
    
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing completed successfully!")

