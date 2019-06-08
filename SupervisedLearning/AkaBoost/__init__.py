# Import statements
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))

# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=1)


# Create model
# base_estimator DecisionTreeClassifier(max_depth=1)

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))

# Fit model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# See accuracy

print('Accuracy score: ', format(accuracy_score(y_test, y_pred)))
print('Precision score: ', format(precision_score(y_test, y_pred)))
print('Recall score: ', format(recall_score(y_test, y_pred)))
print('F1 score: ', format(f1_score(y_test, y_pred)))
