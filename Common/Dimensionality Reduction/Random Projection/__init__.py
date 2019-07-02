# Import statements
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection



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


# Random projection features
rp = random_projection.SparseRandomProjection()
X_rp_train = rp.fit_transform(X_train)
X_rp_test = rp.fit_transform(X_test)
print(X_rp_train.shape)
print(X_train.shape)


# Create model
model = DecisionTreeClassifier()

# Fit model
model.fit(X_rp_train, y_train)

# Predict results
y_rp_pred = model.predict(X_rp_test)

print('Accuracy score: ', format(accuracy_score(y_test, y_rp_pred)))

