from sklearn import datasets, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Process data
iris = datasets.load_iris()

# Fit and predict
# Clustering using kmeans
dbs = DBSCAN()
pred = dbs.fit_predict(iris.data)

print(pred)

# Eval models
rand_score = adjusted_rand_score(iris.target, pred)
sil_score = silhouette_score(iris.data, pred, metric='euclidean')


print("Rand score \nDBSCAN:", rand_score)
print("Silhoutte score \nDBSCAN:", sil_score)


