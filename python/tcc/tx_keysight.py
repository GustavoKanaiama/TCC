import threading
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from bladerf import _bladerf
import os
import subprocess


# Configuração comum
sample_rate = 50e6      # Taxa de amostragem em Hz
gain_tx = 60            # Ganho de TX (-15 a 60 dB)
num_samples_tx = int(20e4) #int(20e6) # Número de amostras para transmissão


# Parâmetros dos pulsos
A0 = 0.8                # Amplitude máxima (0.5 - 8.18mV) (1 - 16.44mV)


# Tipos de Pulso

def gaussian_pulse(t, A0, sigma):
    # Considerando a freq de drive = freq do qubit
    gauss =  A0 * np.exp(-0.5 * ((t)/sigma)**2)
    # if shift == "x": #(I)
    #     gauss =  A0 * np.exp(-0.5 * ((t)/sigma)**2)

    # if shift == "y": #(Q)
    #     t_shift = (1/center_freq)/2
    #     gauss =  A0 * np.exp(-0.5 * ((t - t_shift)/sigma)**2)

    return gauss


# Sweep Gaussian Config

t = np.arange(-num_samples_tx/2, num_samples_tx/2+1) / sample_rate
frame_duration = t[-1] - t[0]


sweep_gaussian = [gaussian_pulse(t, A0, 200e-9)] # number of batches
sweep_freq = [5e9]
num_shots_per_batch = 3 # number of identical shots

# Flags/Counters Globais
flag_finish_capt = False # finish capture one batch
flag_finish_all = False # finish capture all the batches
flag_finish_freq = False # finish to capture all waveforms in that frequency
flag_error = False # check the coherence of the keysight setup and the bladerf setup
flag_ready_to_capture = False # osciloscope needs to wait a signal from blade to be able to read the pulse


# Keysight send config
path_python_310 = r'C:\Users\sampa\AppData\Local\Programs\Python\Python310\python.exe'
path_keysight_script = r'c:\Users\sampa\Desktop\TCC\Infinivision\keysight.py'
path_data_analysis_script = r'c:\Users\sampa\Desktop\TCC\Infinivision\keysight.py'

num_batches = len(sweep_gaussian)
num_freq_sweep = len(sweep_freq)

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
    global flag_finish_all, flag_ready_to_capture, center_freq, flag_finish_capt

    # Gerar amostras IQ para transmitir (senoide simples)

    try:
        for i in range(num_batches):
            for j in range(num_freq_sweep):
                #samples_tx = gaussian_pulse(t, A0, sigma, center_freq, "x")

                samples_tx = sweep_gaussian[i]
                center_freq = sweep_freq[j]

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
                tx_channel = _bladerf.CHANNEL_TX(0)
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

                # Transmitir continuamente até interrupção
                flag_ready_to_capture = True # signal to start the osciloscope setup
                while (running.is_set()):

                    if flag_error:
                        sys.exit(-1)

                    if flag_finish_capt:
                        flag_finish_capt = False
                        print(f"Finish capture - batch {i+1}/{num_batches}, sweep frequency {j+1}/{num_freq_sweep}")

                        if (i == num_batches-1): # finish last batch
                            flag_finish_all = True

                        break
    
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
        flag_finish_all = True


def waveform_capture():
    global flag_finish_capt, flag_error, flag_ready_to_capture
    try:
        for batch in range(num_batches):
            for freq in range(num_freq_sweep):
                flag_finish_capt = False

                cmd = [
                    "powershell.exe",
                    path_python_310,
                    path_keysight_script,
                    str(num_shots_per_batch), str(len(sweep_gaussian)), str(len(sweep_freq)),
                    str(batch),
                    str(center_freq),
                    str(frame_duration)
                ]

                keysight_exit = subprocess.run(cmd)
                exit_code = keysight_exit.returncode

                if exit_code == 1:
                    flag_error = True
                    sys.exit(-1)
                
                flag_finish_capt = True
                flag_ready_to_capture = False
                
                while True: #wait until the signal from blade

                    if flag_ready_to_capture or flag_finish_all:
                        flag_ready_to_capture = False
                        break
                    time.sleep(0.5)
                    

    except Exception as e:
        print(f"Erro inesperado na captura de forma de onda: {e}")
    
    finally:
        print("Finalizando a captura.")

        


# Criar e iniciar as threads de transmissão e recepção
thread_tx = threading.Thread(target=transmit)
thread_waveform_capture = threading.Thread(target=waveform_capture)

thread_tx.start()
thread_waveform_capture.start()

# Manter o código rodando até interrupção do usuário (Ctrl + C)
try:
    while (not flag_finish_all) and (not flag_error):
        time.sleep(1)

except KeyboardInterrupt:
    print("Interrompido pelo usuário.")
    running.clear()  # Certifique-se de que ambos os processos sejam interrompidos
finally:
    thread_tx.join()
    thread_waveform_capture.join()

    sdr.close()  # Garantir o fechamento do dispositivo

    if flag_finish_capt:
        print("Os dados foram armazenados corretamente.")

    print("Dispositivo BladeRF fechado com sucesso")

