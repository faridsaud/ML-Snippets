from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the data from the boston house-prices dataset

boston_data = load_boston()
X = boston_data['data']
y = boston_data['target']

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=1)

# Make and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)



# Make a prediction using the model
y_pred = model.predict(X_test)

print('Mean Absolute score: ', format(mean_absolute_error(y_test, y_pred)))
print('Mean Squared score: ', format(mean_squared_error(y_test, y_pred)))
print('R2 score: ', format(r2_score(y_test, y_pred)))