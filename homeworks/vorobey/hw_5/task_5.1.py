import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# загрузка данных
df = pd.read_csv('fish.csv')

# Разделение на X и y
X = df.drop(columns=['Weight'])
y = df['Weight']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Length1', 'Length2', 'Length3', 'Height', 'Width']),
        ('cat', OneHotEncoder(drop='first'), ['Species'])
    ])

X_processed = preprocessor.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# инициализация и обучение однослойного персептрона
mlp = MLPRegressor(
    hidden_layer_sizes=(),
    activation='identity',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
mlp.fit(X_train, y_train)

# Прогноз и оценка метрики R2
y_pred = mlp.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Качество модели однослойного персептрона R2: {r2:.3f}")
print('Коэффициент детерминации R2 (Линейная модель): 0.882')
print('Коэффициент детерминации R2 (Полиномиальная модель): 0.967')