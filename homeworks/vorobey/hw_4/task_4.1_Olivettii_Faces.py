import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data
y = faces.target

# Кластеризация методом на 40 кластеров
kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Снижение размерности до 2D с помощью t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_2d = tsne.fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# истинное разделение по людям
ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='jet', s=25, alpha=0.8)
ax1.set_title('Исходный датасет Olivetti Faces\n(Истинные классы людей)')
ax1.set_xlabel('t-SNE Компонента 1')
ax1.set_ylabel('t-SNE Компонента 2')
ax1.grid(True, linestyle='--', alpha=0.5)

# как сгруппировал алгоритм
ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='jet', s=25, alpha=0.8)
ax2.set_title('Результаты кластеризации K-Means\n(Выделенные алгоритмом кластеры)')
ax2.set_xlabel('t-SNE Компонента 1')
ax2.set_ylabel('t-SNE Компонента 2')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()