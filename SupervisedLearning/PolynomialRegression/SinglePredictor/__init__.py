from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Assign the data to predictor and outcome variables

train_data = read_csv('data.csv')
X = train_data[['Var_X']].values
y = train_data[['Var_Y']].values

# Create polynomial features
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)


# Make and fit the polynomial regression model
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)