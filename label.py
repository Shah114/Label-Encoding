# Import libraries
import numpy as np
import pandas as pd

# Import dataset
df = pd.read_csv('C:/VS CODE/ML/Encoding/Label/Iris.csv')

print('Unique values of dataset: ')
print(df['Species'].unique())

print('\nValue counts: ')
print(df['Species'].value_counts())

print('\nHead of data:')
print(df.head())

'''Applying Label Encoder'''
# Import label encoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit_transform(df['Species'])

df['Species']= label_encoder.fit_transform(df['Species'])

print('\nUnique values after encoding:')
print(df['Species'].unique())

print('\nValue counts:')
print(df.Species.value_counts())