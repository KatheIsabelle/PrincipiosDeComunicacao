import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy import fftpack
from PIL import Image
import requests
from io import BytesIO

# Carregando imagem
# Definindo endereço da imagem
url = "https://overplay.com.br/wp-content/uploads/2022/01/capa-story-overplay-14-640x853.jpg"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

# Obtendo a imagem
response = requests.get(url, headers=headers)


# Carregando a imagem usando a bilbioteca Pillow
im = Image.open(BytesIO(response.content)).convert('L')


# Convertendo a imagem para um array numpy
im = im.resize((im.width // 4, im.height // 4))
image=np.asarray(im)

# Dimensões da imagem
image_size = image.shape

# Plotando a imagem
plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
plt.imshow(image, cmap = 'gray')
plt.title('Imagem Original')

bit_sequence = np.unpackbits(image.flatten())
print(bit_sequence)

# Configuração de parâmetros
bit_rate = 1000  # Taxa de transmissão de bits em bits por segundo
sample_rate = 50000  # Taxa de amostragem em Hz
carrier_freq = 5000   # Frequência da portadora em Hz (10 kHz)
time_per_bit = 1 / bit_rate  # Tempo por bit
samples_per_bit = int(sample_rate * time_per_bit)  # Número de amostras por bit

# Gerando a sequência de pulsos correspondente aos bits
pulse_train = np.repeat(bit_sequence, samples_per_bit)

# Plotando a sequência de pulsos (amostra pequena para visualização)
plt.figure(figsize=(12, 3))
plt.plot(pulse_train[:10*samples_per_bit], drawstyle='steps-pre')
plt.title('Sequência de Pulsos')
plt.xlabel('Tempo (amostras)')
plt.ylabel('Amplitude')
plt.show()

# Criando o sinal da portadora
t = np.linspace(0, len(pulse_train) / sample_rate, len(pulse_train), endpoint=False)
carrier_signal = np.cos(2 * np.pi * carrier_freq * t)

# Modulação ASK
modulated_signal = pulse_train * carrier_signal

# Plotando o sinal modulado
plt.figure(figsize=(12, 3))
plt.plot(t[:500], modulated_signal[:500])
plt.title('Sinal Modulado em ASK')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.show()

# Simulação do canal [sem ruído]
received_signal = modulated_signal

# Detecção de envoltória usando a transformada de Hilbert
analytic_signal = hilbert(received_signal)
envelope = np.abs(analytic_signal)

# Decodificação dos pulsos
demodulated_bits = envelope > (envelope.max() / 2)  # Threshold para determinar se é '1' ou '0'

# Convertendo de volta para bits originais
received_bits = demodulated_bits[::samples_per_bit]
received_bits = received_bits[:len(bit_sequence)]  # Garantindo o tamanho correto

# Reconstruindo a imagem a partir dos bits recebidos
received_pixels = np.packbits(received_bits)
received_image = received_pixels.reshape(image_size)

# Plotando as imagens original e recebida para comparação
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Recebida")
plt.imshow(received_image, cmap='gray')
plt.axis('off')

plt.show()