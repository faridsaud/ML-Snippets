from sklearn import datasets, preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


# Process data
iris = datasets.load_iris()

normalized_X = preprocessing.normalize(iris.data)


# Fit and predict
# Hierarchical clustering using ward
ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(normalized_X)

# Hierarchical clustering using complete linkage
complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
complete_pred = complete.fit_predict(normalized_X)

# Hierarchical clustering using average linkage
avg = AgglomerativeClustering(n_clusters=3, linkage="average")
avg_pred = avg.fit_predict(normalized_X)


# Eval models
ward_ar_score = adjusted_rand_score(iris.target, ward_pred)

complete_ar_score = adjusted_rand_score(iris.target, complete_pred)

avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print("Scores: \nWard:", ward_ar_score, "\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)
