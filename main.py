
#! ==========================================
#! Imports
#! ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

#! ==========================================
#! Carga de los datos
#! ==========================================

signals_beth = pd.read_csv('./data/bethoven.dat', delimiter=' ', names=['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])
signals_death = pd.read_csv('./data/deathmetal.dat', delimiter=' ', names=['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])
signals_baseline = pd.read_csv('./data/baseline.dat', delimiter=' ', names=['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])

# Extraer las señales EEG
eeg_beth = signals_beth['eeg'].values
eeg_death = signals_death['eeg'].values
eeg_baseline = signals_baseline['eeg'].values

# Gráfico individual para Baseline
plt.figure(figsize=(10, 6))
plt.plot(eeg_baseline, 'r', label='Baseline')
plt.title('Baseline')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(eeg_baseline)])
plt.grid()
plt.legend()
plt.show()

# Gráfico individual para Bethoven
plt.figure(figsize=(10, 6))
plt.plot(eeg_beth, 'b', label='Bethoven')
plt.title('Bethoven')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(eeg_beth)])
plt.grid()
plt.legend()
plt.show()

# Gráfico individual para Deathmetal
plt.figure(figsize=(10, 6))
plt.plot(eeg_death, 'g', label='Deathmetal')
plt.title('Deathmetal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(eeg_death)])
plt.grid()
plt.legend()
plt.show()

#! ==========================================
#! Filtros Temporales
#! ==========================================

# Suavizado con media móvil
windowlength = 10
avgeeg_beth = np.convolve(eeg_beth, np.ones((windowlength,)) / windowlength, mode='same')
avgeeg_death = np.convolve(eeg_death, np.ones((windowlength,)) / windowlength, mode='same')
avgeeg_baseline = np.convolve(eeg_baseline, np.ones((windowlength,)) / windowlength, mode='same')

# Gráfico individual para el suavizado Baseline
plt.figure(figsize=(10, 6))
plt.plot(avgeeg_baseline, 'r', label='Smoothed EEG Baseline')
plt.title('Smoothed EEG Baseline')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(avgeeg_baseline)])
plt.grid()
plt.legend()
plt.show()

# Gráfico individual para el suavizado Bethoven
plt.figure(figsize=(10, 6))
plt.plot(avgeeg_beth, 'b', label='Smoothed EEG Bethoven')
plt.title('Smoothed EEG Bethoven')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(avgeeg_beth)])
plt.grid()
plt.legend()
plt.show()

# Gráfico individual para el suavizado Deathmetal
plt.figure(figsize=(10, 6))
plt.plot(avgeeg_death, 'g', label='Smoothed EEG Deathmetal')
plt.title('Smoothed EEG Deathmetal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(avgeeg_death)])
plt.grid()
plt.legend()
plt.show()

#! ==========================================
#! Filtro Espectral
#! ==========================================

Fs = 512  # Frecuencia de muestreo
lowcut = 1.0  # Frecuencia de corte baja (Hz)
highcut = 50.0  # Frecuencia de corte alta (Hz)

# Función de filtro pasa-banda
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Aplicar el filtro a las señales
filtered_baseline = butter_bandpass_filter(eeg_baseline, lowcut, highcut, Fs, order=6)
filtered_beth = butter_bandpass_filter(eeg_beth, lowcut, highcut, Fs, order=6)
filtered_death = butter_bandpass_filter(eeg_death, lowcut, highcut, Fs, order=6)

# Gráfico individual para la señal filtrada Baseline
plt.figure(figsize=(10, 6))
plt.plot(filtered_baseline, 'r', label='Filtered EEG Baseline')
plt.title('Filtered EEG Baseline')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(filtered_baseline)])
plt.grid()
plt.legend()
plt.show()

# Gráfico individual para la señal filtrada Bethoven
plt.figure(figsize=(10, 6))
plt.plot(filtered_beth, 'b', label='Filtered EEG Bethoven')
plt.title('Filtered EEG Bethoven')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(filtered_beth)])
plt.grid()
plt.legend()
plt.show()

# Gráfico individual para la señal filtrada Deathmetal
plt.figure(figsize=(10, 6))
plt.plot(filtered_death, 'g', label='Filtered EEG Deathmetal')
plt.title('Filtered EEG Deathmetal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.ylim([-800, 800])
plt.xlim([0, len(filtered_death)])
plt.grid()
plt.legend()
plt.show()


#! ==========================================
#! Feature Extraction
#! ==========================================

def optimized_rolling_statistics(data, window_size, label):
    # Convertir datos a matriz de NumPy para operaciones rápidas
    data = np.array(data)
    n = len(data)

    # Inicializar ventanas (usando strides para vectorización)
    strides = np.lib.stride_tricks.sliding_window_view(data, window_size)

    # Calcular estadísticas vectorizadas
    mean = np.mean(strides, axis=1)
    std_dev = np.std(strides, axis=1)
    max_val = np.max(strides, axis=1)
    min_val = np.min(strides, axis=1)
    rms = np.sqrt(np.mean(strides**2, axis=1))

    # Crear DataFrame
    stats = pd.DataFrame({
        "media": mean,
        "desviacion_estandar": std_dev,
        "maximo": max_val,
        "minimo": min_val,
        "RMS": rms,
        "señal": label
    })

    return stats

# Extraer características
features_bethoven = optimized_rolling_statistics(filtered_beth, 256, 0)
features_deathmetal = optimized_rolling_statistics(filtered_death, 256, 1)

# Concatenar datasets
dataset = pd.concat([features_bethoven, features_deathmetal], axis=0)

# Guardar el dataset generado en un archivo CSV
output_file = "./data/features/dataset_features.csv"
dataset.to_csv(output_file, index=False)

#! ==========================================
#! Classifier
#! ==========================================

# Leer el dataset generado
dataset = pd.read_csv("./data/features/dataset_features.csv")

# Preparar los datos
X = dataset.drop(columns=['señal'])  # Variables independientes (características)
y = dataset['señal']  # Variable dependiente (etiqueta de clase)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Visualizar la matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# Calcular el accuracy (exactitud)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Calcular Sensibilidad y Especificidad
tn, fp, fn, tp = cm.ravel()
sensibilidad = tp / (tp + fn)
especificidad = tn / (tn + fp)

print(f"Sensibilidad: {sensibilidad:.4f}")
print(f"Especificidad: {especificidad:.4f}")