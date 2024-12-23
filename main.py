
#! ==========================================
#! Imports
#! ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
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
#! Feature Extraction: Estadísticas
#! ==========================================
def optimized_rolling_statistics(data, window_size, label):
    data = np.array(data)
    strides = np.lib.stride_tricks.sliding_window_view(data, window_size)
    mean = np.mean(strides, axis=1)
    std_dev = np.std(strides, axis=1)
    max_val = np.max(strides, axis=1)
    min_val = np.min(strides, axis=1)
    rms = np.sqrt(np.mean(strides**2, axis=1))

    stats = pd.DataFrame({
        "media": mean,
        "desviacion_estandar": std_dev,
        "maximo": max_val,
        "minimo": min_val,
        "RMS": rms,
        "señal": label
    })
    return stats

features_beth = optimized_rolling_statistics(filtered_beth, 256, 0)
features_death = optimized_rolling_statistics(filtered_death, 256, 1)

#! ==========================================
#! Feature Extraction: Potencias Espectrales
#! ==========================================
def calculate_band_power(signal, fs):
    fft_vals = np.abs(fft(signal))
    fft_freqs = np.fft.fftfreq(len(signal), 1 / fs)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    band_powers = {}
    for band, (low, high) in bands.items():
        band_power = np.sum(fft_vals[(fft_freqs >= low) & (fft_freqs < high)]**2)
        band_powers[f'power_{band}'] = band_power

    return band_powers

# Extraer potencias espectrales
spectral_features_beth = pd.DataFrame([calculate_band_power(filtered_beth[i:i+256], Fs) for i in range(0, len(filtered_beth)-256, 256)])
spectral_features_beth['señal'] = 0  # Etiqueta

spectral_features_death = pd.DataFrame([calculate_band_power(filtered_death[i:i+256], Fs) for i in range(0, len(filtered_death)-256, 256)])
spectral_features_death['señal'] = 1  # Etiqueta

#! ==========================================
#! Concatenar Características
#! ==========================================
features_combined = pd.concat([features_beth, features_death], axis=0)
spectral_combined = pd.concat([spectral_features_beth, spectral_features_death], axis=0)

# Combinar estadísticas y espectrales
final_dataset = pd.concat([features_combined.reset_index(drop=True), spectral_combined.reset_index(drop=True)], axis=1)

# Guardar dataset
final_dataset.to_csv('./data/features/dataset_features.csv', index=False)

#! ==========================================
#! Clasificador
#! ==========================================
# Leer el dataset generado
dataset = pd.read_csv('./data/features/dataset_features.csv')

# Preparar los datos
X = dataset.drop(columns=['señal'])  # Variables independientes
y = dataset['señal']  # Etiquetas

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Validación cruzada
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy promedio (validación cruzada): {np.mean(scores):.4f}")

# Evaluar en conjunto de prueba
y_pred = model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Visualizar matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# Métricas de desempeño
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))