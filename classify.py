import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

colors = ['red', 'green', 'blue']


def plotData(X, y, k, title, labels, centroids):
    for i in range(k):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=labels[i])

    if centroids is not None:
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title(title)
    plt.legend()
    plt.show()


# Load Iris dataset
iris = load_iris()
k = 3
X = iris.data  # Features
y = iris.target  # Target variable
target_names = iris.target_names  # Names of the target classes

plotData(X, y, k, 'Iris dataset classified', target_names, None)


# Unsupervised
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
print("K-means accuracy:", inertia)

plotData(X, labels, k, 'K-means Clustering', target_names, centroids)

# Supervised
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X)

accuracy_svm = accuracy_score(y, y_pred)
print("SVM Accuracy:", accuracy_svm * 100)

plotData(X, y_pred, k, 'SVM classification', target_names, None)
