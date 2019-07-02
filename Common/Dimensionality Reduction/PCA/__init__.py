# Import statements
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def do_PCA(n_components, data):
    pca = PCA(n_components)
    return pca.fit_transform(data)


ds = load_digits()



# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(ds.data),
                                                    ds.target,
                                                    random_state=1)

# All features
# Create model
model = DecisionTreeClassifier()

# Fit model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# See accuracy

print('Accuracy score: ', format(accuracy_score(y_test, y_pred)))


# PCA features
X_PCA_train = do_PCA(10, X_train)
X_PCA_test = do_PCA(10, X_test)
print(X_PCA_train.shape)
print(X_train.shape)


# Create model
model = DecisionTreeClassifier()

# Fit model
model.fit(X_PCA_train, y_train)

# Predict results
y_pred_pca = model.predict(X_PCA_test)

print('Accuracy score: ', format(accuracy_score(y_test, y_pred_pca)))

