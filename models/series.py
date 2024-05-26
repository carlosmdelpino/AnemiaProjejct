import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Descargar y cargar el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/AirQualityUCI.zip"
data = pd.read_csv(url, sep=';', decimal=',', usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], na_values=-200)
data.columns = ['CO_GT', 'PT08_S1_CO', 'NMHC_GT', 'C6H6_GT', 'PT08_S2_NMHC', 'NOx_GT', 'PT08_S3_NOx',
                'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH']

# Preprocesamiento de datos
data.dropna(inplace=True)

# Normalizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Crear secuencias de tiempo para entrenamiento
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 24
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Dividir en conjuntos de entrenamiento y prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(X_train.shape[2]))

model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Función para hacer predicciones a largo plazo
def predict_future(model, data, steps):
    current_seq = data[-SEQ_LENGTH:]
    predictions = []
    for _ in range(steps):
        pred = model.predict(np.array([current_seq]))
        predictions.append(pred[0])
        current_seq = np.vstack([current_seq[1:], pred])
    return np.array(predictions)

# Hacer predicciones a 100 instantes en el futuro
future_predictions = predict_future(model, X_test[-1], 100)

# Invertir la normalización de los datos
future_predictions_rescaled = scaler.inverse_transform(future_predictions)

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(data.index[-100:], data['T'][-100:], label='Real')
plt.plot(pd.date_range(start=data.index[-1], periods=100, freq='H'), future_predictions_rescaled[:, -1], label='Predicción')
plt.xlabel('Fecha')
plt.ylabel('Temperatura')
plt.title('Predicción de 100 instantes en el futuro de la variable T')
plt.legend()
plt.show()
