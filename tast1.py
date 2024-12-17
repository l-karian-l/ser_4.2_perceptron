import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Qt5Agg')

final_df = []
best_loss = float('inf')

# Чтение данных
df = pd.read_csv("input.csv", names=['q1', 'q2', 'psi', 'u1', 'u2', 'tau'])
np.random.shuffle(df.values)

# Разделение на входные и выходные данные
x_train = df[['q1', 'q2']].head(192)
x_test = df[['q1', 'q2']].tail(64)
y_train = df[['psi', 'u1', 'u2']].head(192)
y_test = df[['psi', 'u1', 'u2']].tail(64)

# Нормализация данных
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Создание модели
model = Sequential()
model.add(Dense(256, input_dim=2, use_bias=True))
model.add(Dense(256, activation="sigmoid", use_bias=True))
model.add(Dense(256, activation="sigmoid", use_bias=True))
model.add(Dense(3, input_dim=256, activation="sigmoid", use_bias = True))

# Компиляция модели
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение
model.fit(
    x_train_scaled, y_train_scaled,
    validation_data=(x_test_scaled, y_test_scaled),
    epochs=1000,
    batch_size=7,
    verbose=1
)


# Предсказания и обратная нормализация
predictions_scaled = model.predict(x_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)
final_df = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
final_df['psi_pred'], final_df['u1_pred'], final_df['u2_pred'] = predictions[:, 0], predictions[:, 1], predictions[:, 2]

# Функция для 3D визуализации
def plot_3d_scatter(ax, x, y, z, title, xlabel, ylabel, zlabel, label_actual, label_predicted, z_pred):
    ax.scatter(x, y, z, color='blue', label=label_actual)
    ax.scatter(x, y, z_pred, color='red', label=label_predicted)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()

# Создание 3D графиков
fig = plt.figure(figsize=(15, 10))

# Ψ график
ax1 = fig.add_subplot(131, projection='3d')
plot_3d_scatter(ax1, final_df['q1'], final_df['q2'], final_df['psi'], "Ψ", "q1", "q2", "Ψ", "Фактические", "Прогнозные", final_df['psi_pred'])

# u1 график
ax2 = fig.add_subplot(132, projection='3d')
plot_3d_scatter(ax2, final_df['q1'], final_df['q2'], final_df['u1'], "u1", "q1", "q2", "u1", "Фактические", "Прогнозные", final_df['u1_pred'])

# u2 график
ax3 = fig.add_subplot(133, projection='3d')
plot_3d_scatter(ax3, final_df['q1'], final_df['q2'], final_df['u2'], "u2", "q1", "q2", "u2", "Фактические", "Прогнозные", final_df['u2_pred'])

plt.tight_layout()
plt.show()

# Графики с интерполяцией поверхности
def plot_3d_surface(x, y, z, title, xlabel, ylabel, zlabel):
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

# Интерполяция и отображение
plot_3d_surface(final_df['q1'], final_df['q2'], final_df['psi_pred'], "Ψ Surface", "q1", "q2", "Ψ")
plot_3d_surface(final_df['q1'], final_df['q2'], final_df['u1_pred'], "u1 Surface", "q1", "q2", "u1")
plot_3d_surface(final_df['q1'], final_df['q2'], final_df['u2_pred'], "u2 Surface", "q1", "q2", "u2")

# Вычисление максимальных ошибок
max_psi = (np.max((final_df['psi'] - final_df['psi_pred'])))**2
max_u1 = (np.max((final_df['u1'] - final_df['u1_pred'])))**2
max_u2 = (np.max((final_df['u2'] - final_df['u2_pred'])))**2

#среднеквадратическая ошибка
e = 1/len(final_df)*(max_psi + max_u1+ max_u2)


#loss = model.evaluate(x_test, y_test, verbose=0)[0]

#if loss < best_loss:
#    best_loss = loss
#    best_model = model


# Вывод лучшей ошибки
#print(f"\nЛучшая ошибка (loss): {best_loss:.5f}")


print("max ψ err:", max_psi)
print("max u1 err:", max_u1)
print("max u2 err:", max_u2)

print("max err:", e)
