import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Data
data={'age':[25,30,35,40,45,50,60,70,80],
    'salary':[50000,60000,70000,75000,55000,600000,650000,780000,800000],
    'gender':['Male','Female','Female','Male','Male','Female','Male','Female','Male'],
    'city':['Mumbai','Pune','Delhi','Mumbai','Pune','Hydrabad','Mumbai','Banglore','Pune']
}

df=pd.DataFrame(data)
print("Original Data:")
print(df)

# Step 1: Handling The Missig Values
def handle_missing_values(df):
# Impute Missing Values
    df['age']=df['age'].fillna(df['age'].mean())
    df['salary']=df['salary'].fillna(df['salary'].mean())
    df['gender']=df['gender'].fillna(df['gender'].mode()[0])
    return df

# Step 2: Encoding Categorical Variables
def encode_categorical(df):
# Encode 'gender' Using LabelEncoder
    label_encoder=LabelEncoder()
    df['gender']=label_encoder.fit_transform(df['gender'])
    
# One-hot Encode The 'city' Column
    df=pd.get_dummies(df,columns=['city'],drop_first=True)
    return df

# Step 3: Normalize Numerical Features
def normalize_numerical_features(df):
# Normalize 'age' And 'salary' Using StandardScaler
    scaler=StandardScaler()
    df[['age','salary']]=scaler.fit_transform(df[['age','salary']])
    return df

# Step 4: Feature Engineering
def feature_engineering(df):
# Create A New Feature 'income_per_age' By Dividing Salary by Age
    df['income_per_age']=df['salary']/(df['age']+1e-6)
    return df

# Combine All Steps Into A Pipeline Function
def pipeline(df):
    df=handle_missing_values(df)
    df=encode_categorical(df)
    df=normalize_numerical_features(df)
    df=feature_engineering(df)
    return df

# Apply The Pipeline To The Dataset
processed_df=pipeline(df)

# Display The Processed Data
print("\nProcessed Data:")
print(processed_df)