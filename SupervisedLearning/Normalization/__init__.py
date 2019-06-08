from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read the data.
df = pd.read_csv('data.csv')
print('Age before normalization', df.loc[1]['age'])


# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

normalized_data = pd.DataFrame(data = df)
normalized_data[numerical] = scaler.fit_transform(df[numerical])


print('Age after normalization', normalized_data.loc[1]['age'])

print('Sex before one-hot encoding', normalized_data.loc[1]['sex'])

features_final = pd.get_dummies(normalized_data)


print('Sex is transformed to either Male of Female after one-hot encoding', features_final.loc[1]['sex_ Male'])