from sklearn import datasets, preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


# Process data
iris = datasets.load_iris()

normalized_X = preprocessing.normalize(iris.data)


# Fit and predict
# Clustering using kmeans
kmeans = KMeans(n_clusters=3)
pred = kmeans.fit_predict(normalized_X)

# Eval models
score = adjusted_rand_score(iris.target, pred)
sil_score = silhouette_score(normalized_X, pred, metric='euclidean')

print("Scores: \nKMeans:", score, sil_score)
