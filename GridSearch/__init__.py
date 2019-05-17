from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from Plotting import plot_model

# Read Data
data = np.asarray(pd.read_csv('data.csv', header=None))
X = data[:, 0:2]
y = data[:, 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate different models
decision_tree_model = DecisionTreeClassifier(random_state=42)
svm_model = SVC()
naive_bayes = GaussianNB()

models = [
    {
        "clf": naive_bayes,
        "parameters": {"var_smoothing": [1e-9, 1e-5, 1e-2, 1e-1, 1e-9, 1e-9]},
    },
    {
        "clf": svm_model,
        "parameters": {'C': [0.1, 1, 10, 50], 'kernel': ['poly', 'rbf'], 'degree': [1, 2, 3, 4, 5]},
    },
    {
        "clf": decision_tree_model,
        "parameters": {'max_depth': [2, 4, 6, 8, 10], 'min_samples_leaf': [2, 4, 6, 8, 10],
                       'min_samples_split': [2, 4, 6, 8, 10]},
    }
]


def grid_search(parameters, clf):
    # Make an fbeta_score scoring object.
    scorer = make_scorer(f1_score)

    # Perform grid search on the classifier using 'scorer' as the scoring method.
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    # Fit the grid search object to the training data and find the optimal parameters.
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator.
    best_clf = grid_fit.best_estimator_

    # Fit the new model.
    best_clf.fit(X_train, y_train)

    # Make predictions using the new model.
    best_train_predictions = best_clf.predict(X_train)
    best_test_predictions = best_clf.predict(X_test)

    # Calculate the f1_score of the new model.
    print('The training F1 Score is', f1_score(best_train_predictions, y_train))
    print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

    # Plot the new model.
    plot_model(X, y, best_clf)

    # Let's also explore what parameters ended up being used in the new model.
    print(best_clf)


for model in models:
    print(model)
    grid_search(model['parameters'], model['clf'])
