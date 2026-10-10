import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import auc, roc_auc_score, roc_curve
matplotlib.use("TkAgg")

df = pd.read_csv("heart.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Наивный байесовский классификатор
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_pred = nb_model.predict(X_test_scaled)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f"Наивный Байес Accuracy: {nb_accuracy:.3f}")


# Модель 2: Логистическая регрессия
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Логистическая регрессия Accuracy: {lr_accuracy:.3f}")


# Модель 3: K-ближайших соседей
k_values = range(1, 21)
knn_accuracies = []

best_k = 1
best_knn_accuracy = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    knn_accuracies.append(acc)

    if acc > best_knn_accuracy:
        best_knn_accuracy = acc
        best_k = k

print(
    f"K-ближайших соседей -> Лучшее k: {best_k} с Accuracy: {best_knn_accuracy:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_accuracies, marker="o", linestyle="-", color="b")
plt.title("Зависимость точности KNN от количества соседей (k)")
plt.xlabel("Количество соседей (k)")
plt.ylabel("Accuracy на тестовой выборке")
plt.xticks(k_values)
plt.grid(True)
plt.show()

models = {
    "Наивный Байес": nb_model,
    "Логистическая регрессия": lr_model,
    # Создаем лучшую модель KNN на основе найденного best_k
    "K-ближайших соседей": KNeighborsClassifier(n_neighbors=best_k).fit(
        X_train_scaled, y_train
    ),
}

print("\n--- Подробные метрики качества ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    print(f"\n[ {name} ]")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[idx]
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=cm[i, j],
                va="center",
                ha="center",
                size="xx-large",
                weight="bold",
            )

    ax.set_title(f"Матрица ошибок:\n{name}", fontsize=14, pad=20)
    ax.set_xlabel("Предсказанные классы", fontsize=12)
    ax.set_ylabel("Истинные классы", fontsize=12)
    ax.set_xticklabels(["", "0 (Здоров)", "1 (Болен)"])
    ax.set_yticklabels(["", "0 (Здоров)", "1 (Болен)"])

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 8))

for name, model in models.items():
    # predict_proba возвращает вероятности для каждого класса, столбец [:, 1] (класс 1 - болен)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # площадь под кривой (ROC-AUC)
    roc_auc = auc(fpr, tpr)
    print(f"{name} ROC-AUC: {roc_auc:.4f}")
    plt.plot(
        fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Случайный выбор (AUC = 0.50)")

plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Специфичность)", fontsize=12)
plt.ylabel("True Positive Rate (Чувствительность / Recall)", fontsize=12)
plt.title("ROC-кривые для исследуемых классификаторов", fontsize=14, pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)

plt.show()
