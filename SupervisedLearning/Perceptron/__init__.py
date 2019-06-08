import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read the data.
df = pd.read_csv('data.csv',
                   sep=',',
                   header=None,
                   names=['x1', 'x2', 'y'])


# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['x1', 'x2']],
                                                    df['y'],
                                                    random_state=1)

# create model instance
naive_bayes = Perceptron()

# train model

naive_bayes.fit(X_train, y_train)

# make predictions

y_pred = naive_bayes.predict(X_test)

print('Accuracy score: ', format(accuracy_score(y_test, y_pred)))
print('Precision score: ', format(precision_score(y_test, y_pred)))
print('Recall score: ', format(recall_score(y_test, y_pred)))
print('F1 score: ', format(f1_score(y_test, y_pred)))

