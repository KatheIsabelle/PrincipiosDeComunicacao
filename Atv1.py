import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import librosa
import librosa.display
from IPython.display import Audio

# UNIVERSIDADE FEDERAL DE PELOTAS - ENG. DE COMPUTAÇÃO
# DISCIPLINA: PRINCIPIOS DE COMUNICAÇÃO
# ATV 1 - KATHE ISABELLE


# Filtro Butterworth
def butter_filter(data, cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y

# Endereço do arquivo de áudio
caminho_arquivo = librosa.ex('brahms')

# Leitura do Arquivo
array_audio, taxa_amostragem = librosa.load(caminho_arquivo, sr=None)

# Parâmetros do Filtro
FREQUENCY_CUTOFF = 1200  # Frequência de corte (em Hz)
ORDER = 2  

# Filtro passa-baixa
audio_low = butter_filter(array_audio, FREQUENCY_CUTOFF, taxa_amostragem, 'low', ORDER)

# Filtro passa-alta
audio_high = butter_filter(array_audio, FREQUENCY_CUTOFF, taxa_amostragem, 'high', ORDER)

# Plotar os sinais no domínio do tempo
t = np.linspace(0, len(array_audio) / taxa_amostragem, num=len(array_audio))

plt.figure(figsize=(14, 8))

# S. Original
plt.subplot(3, 1, 1)
plt.plot(t, array_audio, color='blue')
plt.title('Sinal Original')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

# S. passa-baixa (woofer)
plt.subplot(3, 1, 2)
plt.plot(t, audio_low, color='red')
plt.title('Sinal Passa-Baixa (Woofer)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

# S. passa-alta (tweeter)
plt.subplot(3, 1, 3)
plt.plot(t, audio_high, color='orange')
plt.title('Sinal Passa-Alta (Tweeter)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Calcular a FFT dos Sinais
def plot_fft(signal, rate, title):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/rate)
    fft_magnitude = np.abs(np.fft.rfft(signal))
    plt.figure(figsize=(12, 6))
    plt.plot(freq, fft_magnitude)
    plt.title(title)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

# FFT do Sinal original
plot_fft(array_audio, taxa_amostragem, 'FFT do S. Original')

# FFT do Sinal passa-baixa
plot_fft(audio_low, taxa_amostragem, 'FFT do S. Passa-Baixa (Woofer)')

# FFT do Sinal passa-alta
plot_fft(audio_high, taxa_amostragem, 'FFT do S. Passa-Alta (Tweeter)')

# Somar os dois sinais filtrados
audio_reconstructed = audio_low + audio_high

# Sinal reconstruído no domínio do tempo
plt.figure(figsize=(14, 4))
plt.plot(t, audio_reconstructed, color='orange')
plt.title('Sinal Reconstruído (Somado)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.show()

# FFT S. reconstruído
plot_fft(audio_reconstructed, taxa_amostragem, 'FFT S. Reconstruído')

# Reprodução
print("Sinal Original:")
display(Audio(data=array_audio, rate=taxa_amostragem))

print("Sinal Passa-Baixa:")
display(Audio(data=audio_low, rate=taxa_amostragem))

print("Sinal Passa-Alta:")
display(Audio(data=audio_high, rate=taxa_amostragem))

print("Sinal Reconstruído:")
display(Audio(data=audio_reconstructed, rate=taxa_amostragem))
