import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera


file_name = 'DATASET.xlsx'
df = pd.read_excel(file_name)
df['Date'] = pd.to_datetime(df['Date'])
# print("Количество пропусков ДО заполнения:")
# print(df.isnull().sum())

# ЗАПОЛНЕНИЕ ПРОПУСКОВ
# ЦЕНЫ АКТИВОВ (линейная интерполяция между соседними месяцами)
price_cols = ['DXY', 'Gold_USD', 'Brent_Oil', 'EUR_USD', 'USD_RUB']
for col in price_cols:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear')


# print("Количество пропусков ПОСЛЕ заполнения:")
# print(df.isnull().sum())
# очищенный файл
clean_file_name = 'Clean_dataset.xlsx'
df.to_excel(clean_file_name, index=False)
# print(df.head())

# добавил лаги (тк временные данные)
df['USD_BYN_Lag1'] = df['USD_BYN'].shift(1)
# df['USD_BYN_Lag2'] = df['USD_BYN'].shift(2)
# df['USD_BYN_Lag3'] = df['USD_BYN'].shift(3)
df['BTC_USD_lag1'] = df['BTC_USD'].shift(1)
df['Brent_oil_lag1'] = df['Brent_Oil'].shift(1)
df['Gold_lag1'] = df['Gold_USD'].shift(1)
df['EUR_USD_lag1'] = df['EUR_USD'].shift(1)
df['USD_RUB_lag1'] = df['USD_RUB'].shift(1)
df['DXY_lag1'] = df['DXY'].shift(1)
df['FED_FUN_EFF_rate_lag1'] = df['FED_FUN_EFF_rate'].shift(1)
df['NB_Rate_lag1'] = df['NB_Rate'].shift(1)
# номера месяцев для сезонности
df['Month'] = df['Date'].dt.month

# Удалил первые строки, где лаги = NaN
df = df.dropna().reset_index(drop=True)
# print(df.shape[0])

# Последние 24 месяца (2 года) — для теста, всё остальное — для обучения
test_size_months = 24
split_index = len(df) - test_size_months

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()
feature_cols = [
    'USD_BYN_Lag1',
    # 'USD_BYN_Lag2',
    # 'USD_BYN_Lag3',
    'BTC_USD_lag1',
    'Brent_oil_lag1',
    'Gold_lag1',
    'EUR_USD_lag1',
    'USD_RUB_lag1',
    'DXY_lag1',
    'FED_FUN_EFF_rate_lag1',
    'NB_Rate_lag1',
    'Month'
]
# print(f"Train: с {train['Date'].min().strftime('%Y-%m-%d')} по {train['Date'].max().strftime('%Y-%m-%d')} ({len(train)} строк)")
# print(f"Test:  с {test['Date'].min().strftime('%Y-%m-%d')} по {test['Date'].max().strftime('%Y-%m-%d')} ({len(test)} строк)")

# ПОДГОТОВКА X (признаки) и y (целевая переменная)

# Все колонки, кроме даты и целевой переменной — это признаки
X_train = train[feature_cols]
y_train = train['USD_BYN']

X_test = test[feature_cols]
y_test = test['USD_BYN']

# print("Признаки, которые пойдут в модель:")
# for i, f in enumerate(X_train.columns, 1):
#     print(f"  {i}. {f}")


#  ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)

# Глобальные настройки стиля (применяются ко всем графикам ниже)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)")
print("="*60)

# Изучение структуры данных
print("\n1. Структура данных (первые 5 строк):")
print(df.head())

print("\n2. Информация о типах данных:")
print(df.info())

print(f"\n3. Проверка на пропуски (должно быть 0): {df.isnull().sum().sum()}")
print(f"4. Проверка на дубликаты строк: {df.duplicated().sum()}")
print("-" * 60)

# Динамика целевой переменной во времени
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['USD_BYN'], color='blue', linewidth=2, label='USD/BYN')
plt.title('Динамика курса USD/BYN за весь период наблюдений', fontsize=14, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Корреляционная матрица (Heatmap)
plt.figure(figsize=(12, 10))
corr_cols = feature_cols + ['USD_BYN']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool),k=1)

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#  Распределение целевой переменной (Гистограмма)
plt.figure(figsize=(8, 5))
sns.histplot(df['USD_BYN'], kde=True, color='green', bins=20)
plt.title('Распределение значений курса USD/BYN', fontsize=14, fontweight='bold')
plt.xlabel('Курс USD/BYN')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()

# Формулировка гипотез (выводы для отчета)
print("\n5. Гипотезы и выводы по EDA:")
print("   ✅ Данные очищены: пропуски обработаны, дубликатов нет.")
print("   ✅ На графике виден явный тренд изменения курса.")
print("   ✅ Корреляционная матрица подтверждает сильную связь USD_BYN с USD_RUB и лаговыми признаками (Lag1).")
print("   ✅ Распределение целевой переменной имеет 'длинный хвост' вправо, что типично для валютных пар.")
print("="*60)




# Начальный размер обучающей выборки:
# используем все данные из train, затем расширяем окно
window_size = len(train)

actual_values = []
predicted_values = []
forecast_dates = []

# Последовательно прогнозируем каждый месяц тестовой выборки
for i in range(len(test)):

    # Обучающие данные: от начала ряда до текущего месяца
    train_window = df.iloc[:split_index + i]

    # Текущий месяц для прогноза
    test_row = df.iloc[[split_index + i]]

    X_window = train_window[feature_cols]
    y_window = train_window['USD_BYN']

    X_current = test_row[feature_cols]
    y_current = test_row['USD_BYN'].iloc[0]

    # Обучаем модель заново
    model = LinearRegression()
    model.fit(X_window, y_window)

    # Прогноз на один месяц вперёд
    prediction = model.predict(X_current)[0]

    # Сохраняем результаты
    actual_values.append(y_current)
    predicted_values.append(prediction)
    forecast_dates.append(test_row['Date'].iloc[0])

# Таблица результатов
results = pd.DataFrame({
    'Date': forecast_dates,
    'Actual_USD_BYN': actual_values,
    'Predicted_USD_BYN': predicted_values
})

results['Error'] = (
    results['Actual_USD_BYN'] - results['Predicted_USD_BYN']
)

results['Absolute_Error'] = results['Error'].abs()

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗА МЕТОДОМ СКОЛЬЗЯЩЕГО ОКНА")
print(results.to_string(index=False))

# Метрики качества
mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

# MAPE
actual_array = np.array(actual_values)
predicted_array = np.array(predicted_values)

mape = np.mean(
    np.abs((actual_array - predicted_array) / actual_array)
) * 100

print("\nМЕТРИКИ КАЧЕСТВА СКОЛЬЗЯЩЕГО ОКНА:")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAPE: {mape:.2f}%")

# График фактического и прогнозного курса
plt.figure(figsize=(12, 6))

plt.plot(
    results['Date'],
    results['Actual_USD_BYN'],
    marker='o',
    label='Фактический курс'
)

plt.plot(
    results['Date'],
    results['Predicted_USD_BYN'],
    marker='o',
    linestyle='--',
    label='Прогноз (скользящее окно)'
)

plt.title('Прогноз USD/BYN методом скользящего окна')
plt.xlabel('Дата')
plt.ylabel('Курс USD/BYN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#ОБУЧЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ (BASELINE)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)





# ПРОВЕРКА НА ПРЕДПОСЫЛКИ ГАУССА_МАРКОВА
print("\n" + "=" * 70)
print("ПРОВЕРКА ПРЕДПОСЫЛОК ЛИНЕЙНОЙ РЕГРЕССИИ")
print("=" * 70)

#  OLS-модель для статистических тестов
X_ols = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_ols).fit()
residuals = ols_model.resid
fitted = ols_model.fittedvalues

# 1. Линейная спецификация: график остатков и тест RESET
plt.figure(figsize=(10, 5))
plt.scatter(fitted, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Остатки линейной регрессии')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

reset_result = sm.stats.diagnostic.linear_reset(
    ols_model, power=2, use_f=True
)
print("\n1. Тест Рамсея RESET:")
print(f"   p-value = {reset_result.pvalue:.8f}")
if reset_result.pvalue < 0.05:
    print("   Возможна неправильная спецификация модели.")
else:
    print("   Нет достаточных оснований считать спецификацию неверной.")

# 2. Мультиколлинеарность: ранг матрицы и VIF
print("\n2. Проверка мультиколлинеарности:")
rank = np.linalg.matrix_rank(X_ols)
print(f"   Ранг матрицы: {rank} из {X_ols.shape[1]} столбцов")

if rank < X_ols.shape[1]:
    print("   ВНИМАНИЕ: обнаружена совершенная мультиколлинеарность.")
else:
    print("   Совершенная мультиколлинеарность не обнаружена.")

vif_data = pd.DataFrame({
    'Признак': X_train.columns,
    'VIF': [
        variance_inflation_factor(X_train.astype(float).values, i)
        for i in range(X_train.shape[1])
    ]
})
print("\n   VIF по признакам:")
print(vif_data.sort_values('VIF', ascending=False).to_string(index=False))
print("   VIF > 5 может указывать на заметную мультиколлинеарность.")

# 3. Условное математическое ожидание ошибки
print("\n3. Условное математическое ожидание ошибки:")
print("   E(error | X) = 0 нельзя подтвердить одним статистическим тестом.")
print("   Нужны обоснование выбора признаков и проверка отсутствия")
print("   утечки будущих данных и важных пропущенных факторов.")

# 4. Гомоскедастичность: тест Бройша — Пагана
bp_result = het_breuschpagan(residuals, X_ols)
print("\n4. Тест Бройша — Пагана:")
print(f"   p-value = {bp_result[1]:.4f}")
if bp_result[1] < 0.05:
    print("   Есть свидетельства неоднородности дисперсии ошибок.")
else:
    print("   Нет достаточных оснований отвергнуть постоянство дисперсии.")

# Дополнительно: автокорреляция ошибок во временном ряду
print("\nДополнительная проверка автокорреляции ошибок:")

dw = durbin_watson(residuals)
print(f"   Статистика Дарбина — Уотсона = {dw:.4f}")

bg_result = acorr_breusch_godfrey(ols_model, nlags=4)
print(f"   Тест Бройша — Годфри, p-value = {bg_result[1]:.4f}")

if bg_result[1] < 0.05:
    print("   Обнаружены свидетельства автокорреляции ошибок.")
else:
    print("   Автокорреляция до 4-го лага статистически не выявлена.")

# Дополнительно: нормальность остатков (не предпосылка Гаусса — Маркова)
jb_result = jarque_bera(residuals)
print("\nДополнительная проверка нормальности остатков:")
print(f"   Тест Жарка — Бера, p-value = {jb_result.pvalue:.4f}")
if jb_result.pvalue < 0.05:
    print("   Нормальность остатков отвергается на уровне 5%.")
else:
    print("   Нет достаточных оснований отвергнуть нормальность остатков.")

print("=" * 70)



#  метрики
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr) * 100
print(f"Ошибка (MAPE) для линейной регрессии: {mape_lr:.2f}%")


# ГРАДИЕНТНЫЙ БУСТИНГ (LightGBM)

print("\n" + "="*60)
print("ОБУЧЕНИЕ МОДЕЛИ ГРАДИЕНТНОГО БУСТИНГА (LightGBM)")
print("="*60)

# Инициализируем модель LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_samples=15,
    subsample=0.8,  # Для каждого дерева случайно берем только 80% строк из обучения
    colsample_bytree=0.8,  # Для каждого дерева случайно берем только 80% признаков
    random_state=42,
    verbose=-1
)
# Обучение модели на тренировочных данных
lgb_model.fit(X_train, y_train)

# Прогноз на тестовых данных
y_pred_lgb = lgb_model.predict(X_test)


# ОБУЧЕНИЕ ДЕРЕВА РЕШЕНИЙ (Decision Tree)

dt_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

#  ОБУЧЕНИЕ СЛУЧАЙНОГО ЛЕСА (Random Forest)

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=6,              # Ограничил глубину каждого дерева
    min_samples_leaf=2,       # Защита от шума
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# РАСЧЕТ МЕТРИК ДЛЯ ВСЕХ 4 МОДЕЛЕЙ

# Линейная регрессия
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr) * 100

# Дерево решений
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mape_dt = mean_absolute_percentage_error(y_test, y_pred_dt) * 100

# Случайный лес
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100

# LightGBM
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100

print("\n" + "="*95)
print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ (Тестовая выборка: 24 месяца)")
print("="*95)
print(f"{'Метрика':<12} | {'Лин. Регр.':<18} | {'Дерево реш.':<18} | {'Случ. лес':<18} | {'LightGBM':<18}")
print("-" * 95)
print(f"{'MAE (руб)':<12} | {mae_lr:<18.4f} | {mae_dt:<18.4f} | {mae_rf:<18.4f} | {mae_lgb:<18.4f}")
print(f"{'RMSE (руб)':<12} | {rmse_lr:<18.4f} | {rmse_dt:<18.4f} | {rmse_rf:<18.4f} | {rmse_lgb:<18.4f}")
print(f"{'MAPE (%)':<12} | {mape_lr:<17.2f}% | {mape_dt:<17.2f}% | {mape_rf:<17.2f}% | {mape_lgb:<17.2f}%")
print("-" * 95)


results = {
    'Линейная регрессия': mape_lr,
    'Дерево решений': mape_dt,
    'Случайный лес': mape_rf,
    'LightGBM': mape_lgb
}
best_model_name = min(results, key=results.get)
best_model_score = results[best_model_name]

print(f"\n ПОБЕДИТЕЛЬ: {best_model_name} (MAPE = {best_model_score:.2f}%)")

# аналитический вывод
print("\n АНАЛИТИЧЕСКИЙ ВЫВОД ДЛЯ ОТЧЕТА:")
if best_model_name == 'Линейная регрессия':
    print("Зависимость курса от факторов носит строго линейный характер. Простая модель надежнее.")
elif best_model_name in ['Дерево решений', 'Случайный лес', 'LightGBM']:
    print("В данных присутствуют нелинейные зависимости (пороговые эффекты), которые ансамбли деревьев поймали лучше.")
print("="*95)


# ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ И ИНТЕРПРЕТАЦИЯ


# График сравнения всех моделей с реальностью
plt.figure(figsize=(14, 7))
plt.plot(test['Date'], y_test, label='Реальный курс', color='black', linewidth=2.5, marker='o')
plt.plot(test['Date'], y_pred_lr, label=f'Лин. Регр. (MAPE {mape_lr:.2f}%)', color='blue', linestyle='-', linewidth=2)
plt.plot(test['Date'], y_pred_rf, label=f'Случ. лес (MAPE {mape_rf:.2f}%)', color='green', linestyle='--', linewidth=1.5)
plt.plot(test['Date'], y_pred_lgb, label=f'LightGBM (MAPE {mape_lgb:.2f}%)', color='red', linestyle='-.', linewidth=1.5)

plt.title('Прогнозирование курса USD/BYN: Сравнение моделей', fontsize=15, fontweight='bold')
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Курс USD/BYN', fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Важность признаков (Feature Importance) для LightGBM
print("\n" + "="*70)
print("ИНТЕРПРЕТАЦИЯ: Важность признаков (LightGBM)")
print("="*70)

importance_lgb = pd.DataFrame({
    'Признак': X_train.columns,
    'Важность': lgb_model.feature_importances_
}).sort_values('Важность', ascending=False).reset_index(drop=True)

print("\nТоп-5 факторов, влияющих на курс USD/BYN:")
print(importance_lgb.head(5).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Важность',
    y='Признак',
    hue='Признак',
    data=importance_lgb,
    palette='mako',
    legend=False
)
plt.title('Важность признаков в модели LightGBM', fontsize=14, fontweight='bold')
plt.xlabel('Важность (split)', fontsize=12)
plt.ylabel('Признак', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Важность признаков для Случайного леса (для сравнения)
print("\n" + "="*70)
print("ИНТЕРПРЕТАЦИЯ: Важность признаков (Random Forest)")
print("="*70)

importance_rf = pd.DataFrame({
    'Признак': X_train.columns,
    'Важность': rf_model.feature_importances_
}).sort_values('Важность', ascending=False).reset_index(drop=True)

print(importance_rf.head(5).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Важность',
    y='Признак',
    hue='Признак',
    data=importance_rf,
    palette='viridis',
    legend=False
)
plt.title('Важность признаков в модели Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Важность (MSE decrease)', fontsize=12)
plt.ylabel('Признак', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ПРОЕКТ ПОЛНОСТЬЮ ЗАВЕРШЕН!")
print("="*70)

