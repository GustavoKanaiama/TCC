from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Criar uma instância do dispositivo bladeRF
try:
    dev = _bladerf.BladeRF()
except _bladerf.BladeRFError as e:
    print(f"Erro ao abrir o dispositivo bladeRF: {e}")
    exit(1)

# Configurar parâmetros de RX
sample_rate = 10e6       # Taxa de amostragem em Hz
center_freq = 100e6      # Frequência central em Hz
gain = 30                # Ganho (ajuste entre -15 e 60 dB)

# Configurar o canal de recepção (RX)
rx_channel = _bladerf.CHANNEL_RX(0)  # Define o canal RX0 como inteiro

try:
    # Configurar a frequência de RX
    dev.set_frequency(rx_channel, int(center_freq))

    # Configurar a taxa de amostragem de RX
    dev.set_sample_rate(rx_channel, int(sample_rate))

    # Configurar a largura de banda de RX
    dev.set_bandwidth(rx_channel, int(sample_rate / 2))

    # Configurar o ganho de RX
    dev.set_gain(rx_channel, gain)

    # Configurar o modo de ganho (Manual)
    dev.set_gain_mode(rx_channel, 0)  # 0 corresponde ao modo de ganho manual
except _bladerf.BladeRFError as e:
    print(f"Erro ao configurar o dispositivo: {e}")
    dev.close()
    exit(1)

# Configurar o stream síncrono de RX
fmt = _bladerf.Format.SC16_Q11  # Formato das amostras
try:
    dev.sync_config(layout=_bladerf.ChannelLayout.RX_X1,  # RX_X1 para canal único
                    fmt=fmt,
                    num_buffers=16,
                    buffer_size=8192,
                    num_transfers=8,
                    stream_timeout=3500)
except _bladerf.BladeRFError as e:
    print(f"Erro ao configurar o stream: {e}")
    dev.close()
    exit(1)

# Habilitar o módulo de RX
print("Iniciando recepção")
dev.enable_module(rx_channel, True)

# Preparar o gráfico
fig, ax = plt.subplots(figsize=(10, 6))
fft_size = 2048
freq_axis = np.linspace(center_freq - sample_rate / 2, center_freq + sample_rate / 2, fft_size) / 1e6
spectrogram_data = np.zeros((100, fft_size))  # Armazenar as últimas 100 linhas do espectrograma

im = ax.imshow(spectrogram_data, aspect='auto', extent=[freq_axis[0], freq_axis[-1], 0, 10],
               cmap='viridis', origin='lower')

ax.set_xlabel("Frequência [MHz]")
ax.set_ylabel("Tempo [s]")
ax.set_title("Espectrograma em Tempo Real")
fig.colorbar(im, ax=ax, label="Potência [dB]")

# Variáveis para controle
buffer_size = fft_size  # Receber fft_size amostras por vez
bytes_per_sample = 4  # 2 bytes para I e 2 bytes para Q
buf = bytearray(buffer_size * bytes_per_sample)

# Variável para controlar a impressão das primeiras 10 amostras
first_run = True

# Contador de frames para controlar a frequência das impressões
frame_counter = 0
print_interval = 50  # Imprimir a cada 50 frames

# Criar a janela de Hanning
window = np.hanning(fft_size)

# Função de atualização
def update(frame):
    global first_run, frame_counter

    frame_counter += 1  # Incrementar o contador de frames

    # Receber amostras
    try:
        dev.sync_rx(buf, buffer_size)
    except _bladerf.BladeRFError as e:
        print(f"Erro durante a recepção: {e}")
        dev.enable_module(rx_channel, False)
        dev.close()
        exit(1)

    # Converter o buffer para um array numpy de inteiros de 16 bits
    samples = np.frombuffer(buf, dtype=np.int16, count=2 * buffer_size)

    # Separar componentes I e Q e formar números complexos
    samples_complex = samples[0::2] + 1j * samples[1::2]

    # Escalar as amostras para o intervalo -1 a 1
    samples_complex /= 2048.0

    # Imprimir as primeiras 10 amostras complexas (IQ)
    if first_run:
        print("\nPrimeiras 10 Amostras:")
        for i in range(10):
            print(f"Amostra {i+1}: I = {samples_complex[i].real:.4f}, Q = {samples_complex[i].imag:.4f}")
        first_run = False

    # Calcular e exibir o valor máximo da amplitude das amostras a cada 'print_interval' frames
    if frame_counter % print_interval == 0:
        max_amplitude = np.max(np.abs(samples_complex))
        print(f"\nValor Máximo das Amostras: {max_amplitude:.4f}")

        # Avisar se o valor máximo estiver próximo de 1
        if max_amplitude >= 0.9:
            print("Aviso: ADC próximo à saturação. Considere reduzir o ganho.")

    # Aplicar a janela de Hanning
    windowed_samples = samples_complex * window

    # Calcular a FFT usando as amostras com janela
    fft_data = np.fft.fftshift(np.fft.fft(windowed_samples, n=fft_size))
    power_spectrum = 10 * np.log10(np.abs(fft_data) ** 2 + 1e-12)

    # Atualizar o espectrograma
    spectrogram_data[:-1] = spectrogram_data[1:]  # Deslocar os dados existentes
    spectrogram_data[-1] = power_spectrum  # Adicionar os novos dados

    # Atualizar a imagem
    im.set_array(spectrogram_data)

    return [im]

# Criar animação
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

plt.show()

# Desabilitar o módulo de RX e fechar o dispositivo ao fechar a janela
dev.enable_module(rx_channel, False)
dev.close()

