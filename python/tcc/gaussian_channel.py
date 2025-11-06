import threading
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from bladerf import _bladerf

# Configuração comum
sample_rate = 60e6      # Taxa de amostragem em Hz
gain_tx = 50            # Ganho de TX (-15 a 60 dB)
gain_rx1 = 30           # Ganho de RX1 (ajuste entre -15 e 60 dB)
gain_rx2 = 30           # Ganho de RX2 (ajuste entre -15 e 60 dB)
num_samples_tx = int(20e4) # Número de amostras para transmissão 
f_tone_tx = 10e5       # Frequência do tom para transmissão
center_freq = 5e9     # Frequência de transmissão e recepção (1.5 GHz)
fft_size = 2048         # Tamanho da FFT para recepção
alpha = -250e6          # anharmonicity (w01 - w12)


# Parâmetros do pulso Gaussiano
A0 = 0.8              # Amplitude máxima (0.5 - 8.18mV) (1 - 16.44mV)
sigma = 200e-9          # Largura do pulso (5e-9  - 50ns (com overshoot 15.4%), 14e-9 - 80ns (sem overshoot))


_t = np.arange(-num_samples_tx/2, num_samples_tx/2+1) / sample_rate

print("duration aprx: ",_t[-1] - _t[0])

def gaussian_pulse(t, A0, sigma, center_freq, shift):
    # Considerando a freq de drive = freq do qubit

    if shift == "x": #(I)
        gauss =  A0 * np.exp(-0.5 * ((t)/sigma)**2)

    if shift == "y": #(Q)
        t_shift = (1/center_freq)/2
        gauss =  A0 * np.exp(-0.5 * ((t - t_shift)/sigma)**2)

    return gauss

def drag_gauss(t, A0, sigma, center_freq, shift, alpha):

    val_lambda = 0.5 #typ {0.5, 1}

    if shift == "x": #(I)
        gauss =  A0 * np.exp(-0.5 * ((t)/sigma)**2)

    if shift == "y": #(Q)
        t_shift = (1/center_freq)/2
        gauss =  A0 * np.exp(-0.5 * ((t - t_shift)/sigma)**2)

        gauss = val_lambda*(np.gradient(gauss)/alpha)

    return gauss

def gaussian_pulse_tx(t, A0, sigma):

    gauss =  A0 * np.exp(-0.5 * ((t)/sigma)**2)

    return gauss

def step_func(t, A0, perc=1):

    #t_slice = int(floor(len(t)*perc))

    step = np.array([A0 if t_i >= 0 else 0 for t_i in t])

    return step

# Criar uma instância do dispositivo bladeRF
try:
    sdr = _bladerf.BladeRF()
except _bladerf.BladeRFError as e:
    print(f"Erro ao abrir o dispositivo bladeRF: {e}")
    sys.exit(1)

# Variável de controle para encerrar as threads
running = threading.Event()
running.set()

# Função de transmissão (TX)
def transmit():
    try:
        # Gerar amostras IQ para transmitir (senoide simples)
        t = np.arange(-num_samples_tx/2, num_samples_tx/2+1) / sample_rate

        #samples_tx = A0 * np.exp(1j * 2 * np.pi * f_tone_tx * t)
        samples_tx = gaussian_pulse_tx(t, A0, sigma)
        #samples_tx = step_func(t, 0.8)

        samples_scaled = samples_tx * 2047  # Escalar para SC16_Q11 (-2048 a +2047)
        samples_i = np.real(samples_scaled).astype(np.int16)
        samples_q = np.imag(samples_scaled).astype(np.int16)

        # Intercalar amostras I e Q
        interleaved_samples = np.empty(len(samples_i) * 2, dtype=np.int16)
        interleaved_samples[0::2] = samples_i
        interleaved_samples[1::2] = samples_q

        # Converter para bytes
        buf_tx = interleaved_samples.tobytes()

        # Configurar o canal de transmissão (TX)
        tx_channel = _bladerf.CHANNEL_TX(1)
        sdr.set_frequency(tx_channel, int(center_freq))
        sdr.set_sample_rate(tx_channel, int(sample_rate))
        sdr.set_bandwidth(tx_channel, int(sample_rate / 2))
        sdr.set_gain(tx_channel, gain_tx)

        # Configurar o stream síncrono de TX
        sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=16,
                        buffer_size=8192,
                        num_transfers=8,
                        stream_timeout=3500)

        # Habilitar TX
        sdr.enable_module(tx_channel, True)
        print(f"Transmitindo na frequência: {center_freq / 1e6} MHz")
        
        while running.is_set():
            try:
                sdr.sync_tx(buf_tx, num_samples_tx)
            except _bladerf.BladeRFError as e:
                print(f"Erro na transmissão: {e}")
                break
            time.sleep(0)  # Pequeno atraso entre transmissões

    except Exception as e:
        print(f"Erro inesperado na transmissão: {e}")
    finally:
        print("Parando transmissão")
        sdr.enable_module(tx_channel, False)  # Desativa o canal de transmissão imediatamente

# Função de recepção (RX1 e RX2) para mostrar I, Q, RSSI e plotar os sinais recebidos
def receive_and_plot():
    try:
        rx_channel_1 = _bladerf.CHANNEL_RX(0)  # RX1
        rx_channel_2 = _bladerf.CHANNEL_RX(1)  # RX2

        # Configurar RX1
        sdr.set_frequency(rx_channel_1, int(center_freq))
        sdr.set_sample_rate(rx_channel_1, int(sample_rate))
        sdr.set_bandwidth(rx_channel_1, int(sample_rate / 2))
        sdr.set_gain(rx_channel_1, gain_rx1)
        sdr.set_gain_mode(rx_channel_1, _bladerf.GainMode.Manual)

        # Configurar RX2
        sdr.set_frequency(rx_channel_2, int(center_freq))
        sdr.set_sample_rate(rx_channel_2, int(sample_rate))
        sdr.set_bandwidth(rx_channel_2, int(sample_rate / 2))
        sdr.set_gain(rx_channel_2, gain_rx2)
        sdr.set_gain_mode(rx_channel_2, _bladerf.GainMode.Manual)

        # Configurar o stream síncrono para RX1 e RX2
        sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X2,  # Dois canais de RX simultâneos
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=16,
                        buffer_size=8192,
                        num_transfers=8,
                        stream_timeout=3500)

        sdr.enable_module(rx_channel_1, True)
        sdr.enable_module(rx_channel_2, True)
        print("Iniciando recepção")

        # Buffer de recepção intercalada para RX1 e RX2
        buffer_size = fft_size * 2  # Dobrar para os dois canais
        buf_rx = bytearray(buffer_size * 4)  # Buffer para ambos os canais simultâneos

        # Configurar o gráfico
        plt.ion()  # Modo interativo para atualizar o gráfico em tempo real
        fig, (ax1, ax2) = plt.subplots(2, 1)  # Duas subplots, uma para RX1 e outra para RX2
        line_rx1, = ax1.plot(np.zeros(fft_size))
        line_rx2, = ax2.plot(np.zeros(fft_size))
        ax1.set_ylim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax1.set_title("Sinal Recebido no RX1")
        ax2.set_title("Sinal Recebido no RX2")
        ax1.set_xlabel("Amostras")
        ax1.set_ylabel("Amplitude")
        ax2.set_xlabel("Amostras")
        ax2.set_ylabel("Amplitude")

        plt.tight_layout()  # Ajustar o espaçamento entre os subplots

        while running.is_set():
            try:
                # Receber amostras simultâneas de RX1 e RX2
                sdr.sync_rx(buf_rx, buffer_size)
                samples = np.frombuffer(buf_rx, dtype=np.int16)
                samples_i_1 = samples[0::4]  # Amostras I de RX1
                samples_q_1 = samples[1::4]  # Amostras Q de RX1
                samples_i_2 = samples[2::4]  # Amostras I de RX2
                samples_q_2 = samples[3::4]  # Amostras Q de RX2

                # Calcular os sinais complexos para ambos os canais
                samples_complex_1 = samples_i_1 + 1j * samples_q_1
                samples_complex_2 = samples_i_2 + 1j * samples_q_2

                # Calcular RSSI para RX1 e RX2
                rssi_1 = 10 * np.log10(np.mean(np.abs(samples_complex_1) ** 2) + 1e-12)
                rssi_2 = 10 * np.log10(np.mean(np.abs(samples_complex_2) ** 2) + 1e-12)

                # Mostrar os primeiros valores de I, Q, e RSSI para RX1 e RX2
                print(f"RX1 - Primeiro I: {samples_i_1[0]}, Q: {samples_q_1[0]}, RSSI: {rssi_1:.2f} dB")
                print(f"RX2 - Primeiro I: {samples_i_2[0]}, Q: {samples_q_2[0]}, RSSI: {rssi_2:.2f} dB")

                # Atualizar o gráfico com os sinais recebidos
                line_rx1.set_ydata(samples_i_1 / 2048.0)  # Normalizar para -1 a 1
                line_rx2.set_ydata(samples_i_2 / 2048.0)  # Normalizar para -1 a 1
                plt.draw()
                plt.pause(0.01)

            except _bladerf.BladeRFError as e:
                print(f"Erro na recepção: {e}")
                break

    except Exception as e:
        print(f"Erro inesperado na recepção: {e}")
    finally:
        print("Parando recepção")
        sdr.enable_module(rx_channel_1, False)
        sdr.enable_module(rx_channel_2, False)


# Criar e iniciar as threads de transmissão e recepção
thread_tx = threading.Thread(target=transmit)
#thread_rx = threading.Thread(target=receive_and_plot)

thread_tx.start()
#thread_rx.start()

# Manter o código rodando até interrupção do usuário (Ctrl + C)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrompido pelo usuário.")
    running.clear()  # Certifique-se de que ambos os processos sejam interrompidos
finally:
    thread_tx.join()
    #thread_rx.join()
    sdr.close()  # Garantir o fechamento do dispositivo
    print("Dispositivo BladeRF fechado com sucesso")
