import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# installs
# pip install ucimlrepo
# pip install plotly
# pip install seaborn
# pip install scikit-learn
# pip install matplotlib
  
# fetch dataset 
data = fetch_ucirepo(id=53) 
# print(data.data.features)

# grab the iris original csv
iris = data.data['original']
X = data.data.features.values
y = data.data.targets.values

# print(iris)

# print the scatter plot of various features
sns.set_style("whitegrid")
sns.pairplot(iris,hue="class", height=3);
plt.show()

# elbow method
wcss = []

# for k from 1 to 10
for i in range(1, 10):

    # grab the cluster
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(X)

    # append the intertia to wcss
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k = 0
best_silhouette = 0

# for k from 3 to 10
for i in range(3, 10):

    # grab the cluster
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(X)

    # get the class predictions
    y_pred = kmeans.predict(X)

    # get the silhouette score
    silhouette = silhouette_score(X, y_pred)

    print("Silhouette score for k = ", i, ": ", silhouette)

    # if silhouette score is better than the best silhouette score
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        k = i

print("Best silhouette score: ", best_silhouette)
print("Best k: ", k)

# print(data.metadata.preprocessed)

kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)

# fitting the model
kmeans.fit(X)

# getting the class predictions
y_pred = kmeans.predict(X)

#Visualising the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 5, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 5, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 5, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 5, c = 'black', label = 'Centroids')
plt.legend()
plt.show()

fig = px.scatter_3d(iris, x='sepal length', y='sepal width', z='petal length', color='class', symbol='class', size_max=10, opacity=0.7, title="Iris Dataset")
fig.show()

fig = px.scatter_3d(X, x=X[:,0], y=X[:,1], z=X[:,2], color=y_pred, symbol=y_pred, size_max=10, opacity=0.7, title="KMeans Clustering")

# plot the centroids
fig.add_scatter3d(x=kmeans.cluster_centers_[:,0], y=kmeans.cluster_centers_[:,1], z=kmeans.cluster_centers_[:,2], mode='markers', marker=dict(size=10, color='rgb(0,0,0)'))

fig.show()

