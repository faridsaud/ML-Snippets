from sklearn import datasets, preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score


# Process data
iris = datasets.load_iris()

normalized_X = preprocessing.normalize(iris.data)


# Fit and predict
# Clustering using kmeans
gmm = GaussianMixture(n_components=3)
pred = gmm.fit_predict(normalized_X)

# Eval models
rand_score = adjusted_rand_score(iris.target, pred)
sil_score = silhouette_score(normalized_X, pred, metric='euclidean'),


print("Rand score \nGMM:", rand_score)
print("Silhoutte score \nGMM:", sil_score)
