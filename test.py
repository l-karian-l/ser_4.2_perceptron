import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend  

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Qt5Agg')


# Чтение данных
df = pd.read_csv("input.csv", names=['q1', 'q2', 'psi', 'u1', 'u2', 'tau'])
np.random.shuffle(df.values)

# Входные и выходные данные
x_train = df[['q1', 'q2']].head(192)
x_test = df[['q1', 'q2']].tail(64)
y_train = df[['psi', 'u1', 'u2']].head(192)
y_test = df[['psi', 'u1', 'u2']].tail(64)

# Создание модели
model = Sequential()
model.add(Dense(30, input_dim=2, use_bias=True))
model.add(Dense(30, activation="tanh", use_bias=True))
model.add(Dense(30, activation="tanh", use_bias=True))
model.add(Dense(3, input_dim=30, activation="tanh",  use_bias = True))

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=400, batch_size=4, verbose=1)

# Предсказания
input_df = x_test
true_df = y_test
predictions = pd.DataFrame(model.predict(input_df), columns=['psi_pred', 'u1_pred', 'u2_pred'])
predictions.index = x_test.index
final_df = pd.concat([input_df, true_df, predictions], axis=1)

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
max_psi = np.max((final_df['psi'] - final_df['psi_pred'])**2)
max_u1 = np.max((final_df['u1'] - final_df['u1_pred'])**2)
max_u2 = np.max((final_df['u2'] - final_df['u2_pred'])**2)

print("max ψ err:", max_psi)
print("max u1 err:", max_u1)
print("max u2 err:", max_u2)
