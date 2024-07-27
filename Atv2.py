import librosa
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter

# ETAPA 1 IMPORTANDO ÁUDIO [trecho de 5 segundos do áudio]. AUDIO 1 e AUDIO 2

# Endereço do arquivo de áudio
caminho_arquivo = librosa.ex('brahms')
# Leitura do arquivo como um array numpy considerando a taxa de amostragem original do arquivo
array_audio_1, taxa_amostragem_1 = librosa.load(caminho_arquivo, sr=None)

# Retirando um trecho de 5 segundos do áudio
amostras_5s = 5 * taxa_amostragem_1
array_audio_1_cortado=array_audio_1[0:amostras_5s]

# Executando o sinal de áudio
Audio(data=array_audio_1_cortado, rate=taxa_amostragem_1)

# Endereço do arquivo de áudio 2
caminho_arquivo = librosa.ex('trumpet')

# Leitura do arquivo como um array numpy considerando a taxa de amostragem original do arquivo
array_audio_2, taxa_amostragem_2 = librosa.load(caminho_arquivo, sr=None)

# Retirando um trecho de 5 segundos do áudio
amostras_5s = 5 * taxa_amostragem_2
array_audio_2_cortado=array_audio_2[0:amostras_5s]

# Executando o sinal de áudio
Audio(data=array_audio_2_cortado, rate=taxa_amostragem_2)



# ETAPA 2: frequência de amostragem para simular sistema contínuo. Para MSG 1 e MSG 2
# superamostragem (1 MHz) = 1e6
# subamostragem (10 kHz) = 1e4

# Definindo a frequência de amostragem para simular a trasmissão como se tivessemos usando um sistema contínuo
freq_superamostragem = 1e6
freq_subamostragem = 1e4

# Reamostrando o áudio 1 para a frequência mais baixa
array_audio_1_subamostrado = librosa.resample(array_audio_1_cortado, orig_sr=taxa_amostragem_1, target_sr=freq_subamostragem)

# Reamostrando o áudio 1 para a frequência mais alta
array_audio_1_superamostrado = librosa.resample(array_audio_1_subamostrado, orig_sr=freq_subamostragem, target_sr=freq_superamostragem)

#-------------------------------------------------------------

# Reamostrando o áudio 2 para a frequência mais alta
array_audio_2_subamostrado = librosa.resample(array_audio_2_cortado, orig_sr=taxa_amostragem_2, target_sr=freq_subamostragem)

# Reamostrando o áudio 2 para a frequência mais alta
array_audio_2_superamostrado = librosa.resample(array_audio_2_subamostrado, orig_sr=freq_subamostragem, target_sr=freq_superamostragem)



# ETAPA 3: Modulação de 2 sinais ϕ am-dsb (t)

# * A ≥ max(∣m(t)∣), garante deslocamento da parte negativa para cima. 
# Amplitude A deve ser maior ou igual ao valor absoluto máximo da mensagem m(t). 
# Dessa forma, a envoltória do sinal modulado reflete fielmente a mensagem original

# Criar uma portadora [frequência 50 kHz no exemplo]. Necessário criar 2 portadoras diferentes.

# MENSAGEM 1 - PORTADORA 50HZ

# Frequência da portadora da rádio 1 [Hz]
fc1 = 5e4

# Constante A
A = 1.5

# Criando a portadora 1 + valor A
eixo_t=np.linspace(0, 5, len(array_audio_1_superamostrado))
portadora_1=np.zeros([len(array_audio_1_superamostrado)])
for i in range(0,len(array_audio_1_superamostrado)):
    portadora_1[i]=np.cos(2*np.pi*fc1*eixo_t[i])

# Realizando a modulação AM-DSB
mensagem_1_modulada = (A+array_audio_1_superamostrado) * portadora_1

# Plotando o sinal de áudio
plt.figure(figsize = (10, 5))
plt.subplot(3,2,1)
plt.plot(eixo_t,array_audio_1_superamostrado)
plt.ylabel('Amplitude')
plt.title('Mensagem 1')

plt.subplot(3,2,2)
plt.plot(eixo_t,array_audio_1_superamostrado)
plt.xlim(2.6,2.601)


plt.subplot(3,2,3)
plt.plot(eixo_t,portadora_1)
plt.ylabel('Amplitude')
plt.title('Portadora 1')

plt.subplot(3,2,4)
plt.plot(eixo_t,portadora_1)
plt.xlim(2.6,2.601)


plt.subplot(3,2,5)
plt.plot(eixo_t,mensagem_1_modulada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Modulado 1')

plt.subplot(3,2,6)
plt.plot(eixo_t,mensagem_1_modulada)
plt.xlabel('Tempo (s)')
plt.xlim(2.6,2.601)

plt.tight_layout()




# MENSAGEM 2 - PORTADORA 100HZ

# Frequência da portadora da rádio 2 [Hz] 100
fc2 = 10e4

# Constante A
A = 1.5

# Criando a portadora 2
eixo_t=np.linspace(0, 5, len(array_audio_2_superamostrado))
portadora_2=np.zeros([len(array_audio_2_superamostrado)])
for i in range(0,len(array_audio_2_superamostrado)):
    portadora_2[i]=np.cos(2*np.pi*fc2*eixo_t[i])

# Realizando a modulação AM-DSB
mensagem_2_modulada = ( A + array_audio_2_superamostrado) * portadora_2


# Plotando o sinal de áudio
plt.figure(figsize = (10, 5))
plt.subplot(3,2,1)
plt.plot(eixo_t,array_audio_2_superamostrado, color='orange')
plt.ylabel('Amplitude')
plt.title('Mensagem 2')

plt.subplot(3,2,2)
plt.plot(eixo_t,array_audio_2_superamostrado, color='orange')
plt.xlim(2.6,2.601)


plt.subplot(3,2,3)
plt.plot(eixo_t,portadora_2, color='orange')
plt.ylabel('Amplitude')
plt.title('Portadora 2')

plt.subplot(3,2,4)
plt.plot(eixo_t,portadora_2, color='orange')
plt.xlim(2.6,2.601)


plt.subplot(3,2,5)
plt.plot(eixo_t,mensagem_2_modulada, color='orange')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Modulado 2')

plt.subplot(3,2,6)
plt.plot(eixo_t,mensagem_2_modulada, color='orange')
plt.xlabel('Tempo (s)')
plt.xlim(2.6,2.601)

plt.tight_layout()


# ETAPA 4: Analisar a mensagem, portadora e sinal modulado por Transformadas de Fourier [FFT].
# extrair informações das componentes em posições específicas do espectro de frequência dos sinais

# Obtenção da FFT da mensagem 1
fft_array_audio_1_superamostrado = np.fft.fft(array_audio_1_superamostrado)
# Criação do array com frequências para plotar os coeficientes
N = len(fft_array_audio_1_superamostrado)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT da mensagem
plt.figure(figsize = (6, 5))
plt.subplot(3,1,1)
plt.plot(array_freq, np.abs(fft_array_audio_1_superamostrado), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)
plt.title('FFT Mensagem 1')

# Obtenção da FFT da portadora
fft_portadora_1 = np.fft.fft(portadora_1)
# Criação do array com frequências para plotar os coeficientes
N = len(fft_portadora_1)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT da portadora
plt.subplot(3,1,2)
plt.plot(array_freq, np.abs(fft_portadora_1), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)
plt.title('FFT Portadora 1')

# Obtenção da FFT do sinal modulado
fft_mensagem_1_modulada = np.fft.fft(mensagem_1_modulada)

# Criação do array com frequências para plotar os coeficientes
N = len(fft_mensagem_1_modulada)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT do sinal modulado
plt.subplot(3,1,3)
plt.plot(array_freq, np.abs(fft_mensagem_1_modulada), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)
plt.title('FFT Sinal Modulado 1')
plt.tight_layout()

# --------------------------------------------------------------------------------

# Obtenção da FFT da mensagem 2
fft_array_audio_2_superamostrado = np.fft.fft(array_audio_2_superamostrado)

# Criação do array com frequências para plotar os coeficientes
N = len(fft_array_audio_2_superamostrado)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT da mensagem 2
plt.figure(figsize = (10, 5))
plt.subplot(3,1,1)
plt.plot(array_freq, np.abs(fft_array_audio_2_superamostrado), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)
plt.title('FFT Mensagem 2')

# Obtenção da FFT da portadora 2
fft_portadora_2 = np.fft.fft(portadora_2)
# Criação do array com frequências para plotar os coeficientes
N = len(fft_portadora_2)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT da portadora 2
plt.subplot(3,1,2)
plt.plot(array_freq, np.abs(fft_portadora_2), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)
plt.title('FFT Portadora 2')

# Obtenção da FFT do sinal modulado 2
fft_mensagem_2_modulada = np.fft.fft(mensagem_2_modulada)
# Criação do array com frequências para plotar os coeficientes
N = len(fft_mensagem_2_modulada)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT do sinal modulado 2
plt.subplot(3,1,3)
plt.plot(array_freq, np.abs(fft_mensagem_2_modulada), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)
plt.title('FFT Sinal Modulado 2')
plt.tight_layout()


# ETAPA 5: Criação de Canal
# Os sinais modulados são somados para simular a transmissão em um canal compartilhado ideal.
# Resultado do compartilhamento do canal utilizando o gráfico do sinal no domínio do tempo e sua 
#Transformada de Fourier. No domínio do tempo os sinais foram misturados, parecendo serem 
#indicerníveis. Porém, no domínio da Transformadade Fourier, as componentes estão separadas, 
# sendo possível recuperar os sinais por meio de um processo de demodulação.

canal=mensagem_1_modulada+mensagem_2_modulada

# Obtenção da FFT da mensagem
fft_canal = np.fft.fft(canal)
# Criação do array com frequências para plotar os coeficientes
N = len(fft_canal)
n = np.arange(N)
T = N/freq_superamostragem
array_freq = n/T

# Plotando os coeficientes da FFT da mensagem
plt.figure(figsize = (10, 5))
plt.plot(array_freq, np.abs(fft_canal), 'k')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude FFT |X(f)|')
plt.xlim(0, freq_superamostragem/2)

# Plotando o sinal de áudio
plt.figure(figsize = (10, 5))
plt.plot(eixo_t,canal)
plt.ylabel('Amplitude')


array_audio_canal = librosa.resample(canal, orig_sr=freq_superamostragem, target_sr=taxa_amostragem_1)

# Executando o sinal de áudio
Audio(data=array_audio_canal, rate=taxa_amostragem_1)



# ETAPA 6: Demodulação Síncrona
# Sinal do canal é multiplicado por uma portadora, 
#com frequência igual à da portadora usada na modulação, 
#e é posteriormente filtrado utilizando um filtro passa-baixas.

# MENSAGEM 1 

# Filtro passa-faixas para a demodulação
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parâmetros do filtro passa-faixas
lowcut = fc1 - 5000
highcut = fc1 + 5000

# Aplicando o filtro passa-faixas
mensagem_1_filtrada = bandpass_filter(mensagem_1_modulada, lowcut, highcut, freq_superamostragem)

# Retificação do sinal para detecção de envoltória
mensagem_1_retificada = np.abs(mensagem_1_filtrada)

# Parâmetros do filtro passa-baixas para a detecção de envoltória
cutoff = 10000
b, a = butter(6, cutoff, btype='low', fs=freq_superamostragem)
mensagem_1_envoltoria = filtfilt(b, a, mensagem_1_retificada)

# Bloqueador de valor DC (subtrair a média do sinal)
mensagem_1_demodulada = mensagem_1_envoltoria - np.mean(mensagem_1_envoltoria)

# Resampling
array_audio_demodulado_1 = librosa.resample(mensagem_1_demodulada, orig_sr=freq_superamostragem, target_sr=taxa_amostragem_1)

# Executando o sinal de áudio demodulado
Audio(data=array_audio_demodulado_1, rate=taxa_amostragem_1)

# Plotando os sinais
plt.figure(figsize=(12, 10))

# Sinal modulado
plt.subplot(4, 1, 1)
plt.plot(eixo_t, mensagem_1_modulada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Modulado')

# Sinal filtrado
plt.subplot(4, 1, 2)
plt.plot(eixo_t, mensagem_1_filtrada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Filtrado (Passa-Faixas)')

# Sinal retificado
plt.subplot(4, 1, 3)
plt.plot(eixo_t, mensagem_1_retificada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Retificado')

# Sinal demodulado (Envoltória)
plt.subplot(4, 1, 4)
plt.plot(eixo_t, mensagem_1_demodulada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Demodulado (Envoltória)')

plt.tight_layout()
plt.show()


# MENSAGEM 2


# Filtro passa-faixas para a demodulação
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parâmetros do filtro passa-faixas
lowcut = fc2 - 5000
highcut = fc2 + 5000

# Aplicando o filtro passa-faixas
mensagem_2_filtrada = bandpass_filter(mensagem_2_modulada, lowcut, highcut, freq_superamostragem)

# Retificação do sinal para detecção de envoltória
mensagem_2_retificada = np.abs(mensagem_2_filtrada)

# Parâmetros do filtro passa-baixas para a detecção de envoltória
cutoff = 10000
b, a = butter(6, cutoff, btype='low', fs=freq_superamostragem)
mensagem_2_envoltoria = filtfilt(b, a, mensagem_2_retificada)

# Bloqueador de valor DC (subtrair a média do sinal)
mensagem_2_demodulada = mensagem_2_envoltoria - np.mean(mensagem_2_envoltoria)

# Resampling
array_audio_demodulado_2 = librosa.resample(mensagem_2_demodulada, orig_sr=freq_superamostragem, target_sr=taxa_amostragem_2)

# Executando o sinal de áudio demodulado
Audio(data=array_audio_demodulado_2, rate=taxa_amostragem_2)

# Plotando os sinais
plt.figure(figsize=(12, 10))

# Sinal modulado
plt.subplot(4, 1, 1)
plt.plot(eixo_t, mensagem_2_modulada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Modulado 2')

# Sinal filtrado
plt.subplot(4, 1, 2)
plt.plot(eixo_t, mensagem_2_filtrada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Filtrado (Passa-Faixas) 2')

# Sinal retificado
plt.subplot(4, 1, 3)
plt.plot(eixo_t, mensagem_2_retificada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Retificado 2')

# Sinal demodulado (Envoltória)
plt.subplot(4, 1, 4)
plt.plot(eixo_t, mensagem_2_demodulada)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Demodulado (Envoltória) 2')

plt.tight_layout()
plt.show()
