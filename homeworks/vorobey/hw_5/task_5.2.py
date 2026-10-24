import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# на всякий
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# папки train и test лежат прямо с файлом кода
TRAIN_PATH = os.path.join(BASE_DIR, 'train')
TEST_PATH = os.path.join(BASE_DIR, 'test')


# ТРАНСФОРМАЦИИ ДЛЯ ПРЕОБРАЗОВАНИЯ ИЗОБРАЖЕНИЙ
transform = transforms.Compose([
    # преобразование в оттенки серого (один канал)
    transforms.Grayscale(num_output_channels=1),
    # изменение размеров (к квадрату 32x32)
    transforms.Resize((32, 32)),
    # конвертация PIL Image/numpy массива в torch.Tensor
    transforms.ToTensor(),
    # стандартизация для 1 канала (чтобы избежать ошибок выполнения)
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    ),
    # раскатка тензора [C, H, W] в одномерный вектор [C * H * W]
    transforms.Lambda(lambda x: torch.flatten(x))
])


# ЗАГРУЗКА ДАННЫХ ИЗ ЛОКАЛЬНОЙ ДИРЕКТОРИИ В TORCH
# Создание объектов Dataset по новым путям
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)

# Создание загрузчиков DataLoader для пакетной обработки
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,  # количество изображений в пакете
    shuffle=True,  # обязательное перемешивание для обучающей выборки
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,  # тестовую выборку перемешивать не нужно
)

# Проверка сопостовления классов
print("Индексы классов:", train_dataset.class_to_idx)
categories = list(train_dataset.class_to_idx.keys())


# АРХИТЕКТУРА ОДНОСЛОЙНОГО ПЕРСЕПТРОНА
class SingleLayerPerceptron(nn.Module):
    def __init__(self, input_dim=1024, output_dim=2):
        super(SingleLayerPerceptron, self).__init__()
        # Чистый персептрон: один линейный слой
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Инициализация модели, функция потерь и оптимизатор
model = SingleLayerPerceptron(input_dim=1024, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ЦИКЛ ОБУЧЕНИЯ МОДЕЛИ

epochs = 15
print("\nНачало обучения персептрона...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Сброс градиентов
        outputs = model(images)  # Прямой проход (forward)
        loss = criterion(outputs, labels)  # Расчет ошибки
        loss.backward()  # Обратный проход (backward)
        optimizer.step()  # Обновление весов

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Эпоха [{epoch + 1}/{epochs}], Потери (Loss): {epoch_loss:.4f}")


# ОЦЕНКА ВЕРНОСТИ И CONFUSION MATRIX
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Выбираем класс с наибольшей вероятностью

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Расчет показателя верности (Accuracy)
accuracy = np.mean(all_preds == all_labels)
print(f"\nПоказатель верности (Accuracy) на тестовой выборке: {accuracy:.4f}")

# Построение и визуализация матрицы ошибок (Confusion Matrix)
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
plt.show()
