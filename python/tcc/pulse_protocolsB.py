import threading
import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from bladerf import _bladerf
from datetime import datetime
import subprocess
from tqdm import tqdm

# Absolute Paths - Keysight Device
path_python_310 = r'C:\Users\sampa\AppData\Local\Programs\Python\Python310\python.exe'
path_keysight_setup_script = r'c:\Users\sampa\Desktop\TCC\Infinivision\keysight_setup.py'
path_keysight_request_data_script = r'c:\Users\sampa\Desktop\TCC\Infinivision\keysight_request_data.py'


# ──────────────────────────────────────────────────────────────
#  ▒▒  Parâmetros “fixos” do Sweep
# ──────────────────────────────────────────────────────────────
Fs_SWEEP        = 5e6               # taxa de amostragem (3 MS/s)
NSAMP_SWEEP     = 2048              # amostras por bloco/canal (FFT_SIZE do monitor)
NBLK            = 64                # blocos válidos agregados por frequência
N_THROW         = 1                 # blocos descartados logo após mudar freq
F_TONE_SWEEP    = 40e4              # tom transmitido (KHz)
GAIN_TX0        = 35                # ganho inicial de TX
GAIN_TX1        = 35
GAIN_RX0        = 30                # ganho RX0
GAIN_RX1        = 30                # ganho RX1
FSTART_SWEEP    = 2.3e9             # start do sweep
FSTOP_SWEEP     = 2.9e9             # fim do sweep
FSTEP_SWEEP     = 25e6              # step do sweep
WINDOW          = 'bharris'         # ou 'hanning' janela FFT p/ S21 (Blackman-Harris ou Hanning)
METHOD          = 'mean'            # ou 'median', agregação (median é mais robusto a outlier)
DWELL           = 0.1               # tempo de settling após set_frequency (s)
MIN_REF_BIN     = 1e-9              # |Fref[k]| mínimo para aceitar bloco (evita div/0)
S2P_FILE        = './sweep/s21.s2p'
RATIO_METHOD    = 'mean'
N_CALL_SWEEP = NSAMP_SWEEP*2                   # nº “samples” a pedir no sync_rx
k_bin  = int(round(F_TONE_SWEEP*NSAMP_SWEEP/Fs_SWEEP))

running   = threading.Event()

# Configuração Protocolo de Pulso
sample_rate         = 20e6          # Taxa de amostragem em Hz
num_samples_tx      = int(2**20)    # Número de amostras para transmissão - 2^20 = 1e6 aprx 
N_SHOTS_PULSE       = 10
pulse_center_freq   = None           # Frequência de TX/RX
delta_t_samples     = 150 # Delta time (drive - ressonator) in number of samples

N_CALL_PULSE = num_samples_tx*2

# Parâmetros do pulso Gaussiano
A0 = 0.8                # Amplitude máxima (0.5 - 8.18mV) (1 - 16.44mV)
sigma = 200e-9          # Largura do pulso (5e-9  - 50ns (com overshoot 15.4%), 14e-9 - 80ns (sem overshoot))

# TX1 -> ressonador
# TX2 -> osciloscopio
# RX1 -> saída do Div de potência (ref)
# RX2 -> volta do ressonador (measure)

mag_plot = None
freq_plot = None
phase_deg_plot = None

def gauss_pulse(t, t_samples_shift, A0, sigma):
    # t_samples_shift: delta time (drive - ressonator) in number of samples

    # Convert number of samples to time
    T1 = t[1]-t[0]
    t_shift = t_samples_shift * T1

    gauss = A0*np.exp(-0.5*((t-t_shift)/sigma)**2)
    return gauss

def step_func(t, t_samples_shift, A0, perc=1):
    #t_slice = int(floor(len(t)*perc))
    T1 = t[1]-t[0]
    t_shift = t_samples_shift * T1

    step = []
    ref = perc * ((num_samples_tx/2)/sample_rate)

    for t_i in t:

        if (t_i < (ref)) and (t_i>=0):
            step.append(A0)

        else:
            step.append(0)

    step = np.array(step)
    return step


# ──────────────────────────────────────────────────────────────
#  ▒▒  Cabeçalho S2P
# ──────────────────────────────────────────────────────────────
with open(S2P_FILE, 'w') as f:
    f.write(f"! S21 sweep bladeRF  {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"Frequency [MHz], |S21| [db], Phase [deg]\n")

_t = np.arange(-num_samples_tx/2, num_samples_tx/2+1) / sample_rate

s21_list = [] # List of tuples (Frequency, Mag_db, Phase_deg)

# Control Flags
flag_finish_transmit = False
flag_finish_capt = False
flag_ready_to_capture = False
flag_finish_sweep = False

# Calculation Functions
def rssi_block_dbfs(i, q):
    """↦ RSSI (dBFS) do bloco, mesma fórmula do monitor."""
    p = np.mean(i.astype(np.float64)**2 + q.astype(np.float64)**2)
    return 10*np.log10(p/(2047.0**2) + 1e-12)

def blackman_harris(N):
    n = np.arange(N)
    return (0.35875
            - 0.48829*np.cos(2*np.pi*n/(N-1))
            + 0.14128*np.cos(4*np.pi*n/(N-1))
            - 0.01168*np.cos(6*np.pi*n/(N-1)))
W = blackman_harris(NSAMP_SWEEP) if WINDOW == 'bharris' else np.hanning(NSAMP_SWEEP)

def measure_rssi_quick(sdr, rx_ref, buf, n_call, n_avg=4):
    """medição rápida do RSSI_ref (para o servo)"""
    vals=[]
    for _ in range(n_avg):
        sdr.sync_rx(buf, n_call)
        d=np.frombuffer(buf,np.int16)
        vals.append(rssi_block_dbfs(d[0::4], d[1::4]))
    return float(np.mean(vals))


def s21_bins(ref_iq, meas_iq, k):
    """↦ bin k da FFT de cada canal"""
    return np.fft.fft(ref_iq)[k], np.fft.fft(meas_iq)[k]

# ──────────────────────────────────────────────────────────────
#  ▒▒  Thread TX: gera tom contínuo I/Q
# ──────────────────────────────────────────────────────────────
def tx_sweep(sdr, tx_ch):
    t = np.arange(NSAMP_SWEEP) / Fs_SWEEP
    sig = 0.9 * np.exp(1j*2*np.pi*F_TONE_SWEEP*t)
    i16 = np.round(sig.real*2047).astype(np.int16)
    q16 = np.round(sig.imag*2047).astype(np.int16)
    inter = np.empty(NSAMP_SWEEP*2, np.int16); inter[0::2],inter[1::2] = i16,q16
    payload = inter.tobytes()

    sdr.set_sample_rate(tx_ch, int(Fs_SWEEP))
    sdr.set_bandwidth(tx_ch,  int(Fs_SWEEP/2))
    sdr.set_gain(tx_ch, GAIN_TX0)
    sdr.sync_config(_bladerf.ChannelLayout.TX_X1,
                    _bladerf.Format.SC16_Q11,16,NSAMP_SWEEP,8,10000)
    sdr.enable_module(tx_ch,True)
    try:
        while running.is_set() and not flag_finish_sweep:
            sdr.sync_tx(payload, NSAMP_SWEEP, 10000)
    finally:
        sdr.enable_module(tx_ch,False)
        print("Finish TX Sweep loop")

# Função de transmissão (TX)
def tx_pulse(sdr, pulse_ch, ress_ch):
    try:
        # --------------- DRIVE/RESSONATOR Pulse ----------------------
        # Setup (Gerar amostras IQ para transmitir)
        t = np.arange(-num_samples_tx/2, num_samples_tx/2+1) / sample_rate

        # Drive Pulse
        samples_tx0 = gauss_pulse(t=t, t_samples_shift=0, A0=1, sigma=400e-9)

        # Ressonator Pulse (time shifted)
        # samples_tx0 = gauss_pulse(t=t, t_samples_shift=delta_t_samples, A0=1, sigma=400e-9)*np.exp(2*np.pi*1j*pulse_center_freq*t)
        samples_tx1 = gauss_pulse(t=t, t_samples_shift=delta_t_samples, A0=1, sigma=400e-9)

        samples_scaled_tx0 = samples_tx0 * 2047  # Escalar para SC16_Q11 (-2048 a +2047)
        samples_scaled_tx1 = samples_tx1 * 2047  # Escalar para SC16_Q11 (-2048 a +2047)

        samples_i0 = np.real(samples_scaled_tx0).astype(np.int16)
        samples_q0 = np.imag(samples_scaled_tx0).astype(np.int16)
        
        samples_i1 = np.real(samples_scaled_tx1).astype(np.int16)
        samples_q1 = np.imag(samples_scaled_tx1).astype(np.int16)

        # Intercalar amostras I e Q
        interleaved_samples = np.empty(len(samples_i0) * 4, dtype=np.int16)

        interleaved_samples[0::4] = samples_i0
        interleaved_samples[1::4] = samples_q0
        interleaved_samples[2::4] = samples_i1
        interleaved_samples[3::4] = samples_q1

        # Converter para bytes
        buf_tx_drive = interleaved_samples.tobytes()

        # --------------- Configuração do Canal ----------------

        # Configurar o canal de transmissão
        sdr.set_frequency(ress_ch, int(pulse_center_freq))
        sdr.set_sample_rate(ress_ch, int(sample_rate))
        sdr.set_bandwidth(ress_ch, int(sample_rate / 2))
        sdr.set_gain(ress_ch, GAIN_TX1)

        # Configurar o canal de transmissão
        sdr.set_frequency(pulse_ch, int(pulse_center_freq))
        sdr.set_sample_rate(pulse_ch, int(sample_rate))
        sdr.set_bandwidth(pulse_ch, int(sample_rate / 2))
        sdr.set_gain(pulse_ch, GAIN_TX0)

        # Configurar o stream síncrono de TX
        sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X2,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=16,
                        buffer_size=8192,
                        num_transfers=8,
                        stream_timeout=3500)

        T1 = t[1]-t[0]
        t_dephase = (t[1]-t[0]) * delta_t_samples

        print(f"Time shift of 1 Sample: {T1:.2e}")
        print(f"Total time shift: {t_dephase:.2e}")

        sdr.enable_module(pulse_ch, True)
        sdr.enable_module(ress_ch, True)
        # Transmitir continuamente até interrupção
        print(f"Transmitindo Pulso DRIVE na frequência: {pulse_center_freq / 1e6} MHz")
        while running.is_set():
            try:
                sdr.sync_tx(buf_tx_drive, num_samples_tx*2) # *2 pois utiliza os dois canais (interleaved_samples de dois canais aumenta o buf_tx)
                time.sleep(0)

            except _bladerf.BladeRFError as e:
                print(f"Erro na transmissão: {e}")
                break

        sdr.enable_module(pulse_ch, False)  # Desativa o canal de transmissão imediatamente
        sdr.enable_module(ress_ch, False)

    except Exception as e:
        print(f"Erro inesperado na transmissão: {e}")
    finally:
        print("Parando transmissão")



# ──────────────────────────────────────────────────────────────
#  ▒▒  Thread RX: faz o sweep de frequência
# ──────────────────────────────────────────────────────────────
def rx_sweep(sdr, tx_ch, rx_ref, rx_meas):
    global txlut, flag_finish_sweep, rx_pulse_freq

    # ⚑ 1. Configura ambos RX em modo Manual
    for ch,g in ((rx_ref,GAIN_RX0),(rx_meas,GAIN_RX1)):
        sdr.set_sample_rate(ch,int(Fs_SWEEP))
        sdr.set_bandwidth(ch,int(Fs_SWEEP/2))
        sdr.set_gain(ch,g)
        sdr.set_gain_mode(ch,_bladerf.GainMode.Manual)

    sdr.sync_config(_bladerf.ChannelLayout.RX_X2,
                    _bladerf.Format.SC16_Q11,16,NSAMP_SWEEP,8,10000)
    sdr.enable_module(rx_ref,True); sdr.enable_module(rx_meas,True)

    # ⚑ 2. Pré-aloca buffers
    freqs = np.arange(FSTART_SWEEP, FSTOP_SWEEP+FSTEP_SWEEP, FSTEP_SWEEP)
    buf_rx = bytearray(NSAMP_SWEEP*2*4)      # I/Q de dois canais

    # ganho atual de TX
    g_tx = GAIN_TX0

    # Arquivos auxiliares
    with open(S2P_FILE,'a') as f_s2p,\
         tqdm(freqs,ncols=100,
              desc=('S21 sweep (TXLUT)')) as bar:


        # ⚑ 3. Loop principal sobre as frequências
        for f_rf in bar:
            # 3a. sintoniza LO único (TX + RX0 + RX1)
            for ch in (tx_ch, rx_ref, rx_meas):
                sdr.set_frequency(ch, int(f_rf))
            time.sleep(DWELL)

            # 3b. descarta blocos iniciais (N_THROW) para purgar transiente
            for _ in range(N_THROW):
                sdr.sync_rx(buf_rx, N_CALL_SWEEP)

            # 3d. descarta mais dois blocos após mudar TxGain
            for _ in range(2): sdr.sync_rx(buf_rx, N_CALL_SWEEP, 10000)

            # ⚑ 4. Coleta NBLK blocos “válidos”
            ratios   = []
            for _ in range(NBLK):
                sdr.sync_rx(buf_rx, N_CALL_SWEEP, 10000)
                d = np.frombuffer(buf_rx, np.int16)

                ir, qr = d[0::4], d[1::4]         # RX0
                im, qm = d[2::4], d[3::4]         # RX1

                ref_iq  = (ir+1j*qr) * W
                meas_iq = (im+1j*qm) * W
                Fref_k, Fmeas_k = s21_bins(ref_iq, meas_iq, k_bin)
                if abs(Fref_k) > MIN_REF_BIN:
                    ratios.append(Fmeas_k / Fref_k)

            # 4b. Agrega S21
            if ratios:
                ratio_val = (np.median if METHOD=='median' else np.mean)(ratios)
            else:
                ratio_val = np.nan+1j*np.nan
            mag_db = 20*np.log10(abs(ratio_val)+1e-12)
            phase_deg = np.angle(ratio_val, deg=True)

            # 4c. Console + arquivos
            bar.write(f"{f_rf/1e6:7.2f} MHz |S21|={mag_db:6.2f} dB  "
                      f"TX={g_tx} dB")
            f_s2p.write(f"{f_rf:.6e} {mag_db:.6f} {phase_deg:.6f}\n")
            s21_list.append((f_rf, mag_db, phase_deg))

        flag_finish_sweep = True

    print("Finish RX Sweep loop")
    sdr.enable_module(rx_ref, False)
    sdr.enable_module(rx_meas, False)
    running.clear()


# Função de recepção (RX1 e RX2) para mostrar I, Q, RSSI e plotar os sinais recebidos
def rx_pulse(sdr, rx_ref, rx_meas):
    global mag_plot, freq_plot, phase_deg_plot
    # --Pré-alocar o buffer de agreg ---

    try:
        # Configurar RX0 (ref)
        sdr.set_frequency(rx_ref, int(pulse_center_freq))
        sdr.set_sample_rate(rx_ref, int(sample_rate))
        sdr.set_bandwidth(rx_ref, int(sample_rate / 2))

        # Configurar RX1 (meas)
        sdr.set_frequency(rx_meas, int(pulse_center_freq))
        sdr.set_sample_rate(rx_meas, int(sample_rate))
        sdr.set_bandwidth(rx_meas, int(sample_rate / 2))


        # Configurar o stream síncrono para RX_X2
        sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X2,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=16,
                        buffer_size=8192,
                        num_transfers=8,
                        stream_timeout=10000)

        sdr.enable_module(rx_ref, True)
        sdr.enable_module(rx_meas, True)
        print("RX: Iniciando recepção do Pulso e cálculo S21/S11")

        W_pulse = blackman_harris(num_samples_tx) if WINDOW == 'bharris' else np.hanning(num_samples_tx)

        # O buffer de recepção DEVE corresponder ao tamanho do pulso TX
        buffer_size_per_channel = num_samples_tx 
        
        # Buffer total para I0, Q0, I1, Q1 -> 4 * num_samples_tx * (2 bytes/amostra int16)
        buf_rx = bytearray(int(buffer_size_per_channel * 4 * 2))

        ratios_pulse   = []
        ratio_plot = []
        k_bin_pulse  = 0 #int(round(pulse_center_freq*num_samples_tx/sample_rate))

        for _ in range(20):
            try:
                sdr.sync_rx(buf_rx, buffer_size_per_channel*2)
                d = np.frombuffer(buf_rx, np.int16)

                ir, qr = d[0::4], d[1::4]         # RX0
                im, qm = d[2::4], d[3::4]         # RX1

                ref_iq  = (ir+1j*qr) * W_pulse
                meas_iq = (im+1j*qm) * W_pulse

                F_ref  = np.fft.fft(ref_iq)[k_bin_pulse]
                F_meas = np.fft.fft(meas_iq)[k_bin_pulse]

                if abs(F_ref) > MIN_REF_BIN:
                    ratios_pulse.append(F_meas / F_ref)
                
                ratio_plot.append(np.fft.fft(ref_iq) / (np.fft.fft(meas_iq)))

            except Exception as e:
                print(f"Erro inesperado na recepção: {e}")
                break
                
        # Interuupts the tx_pulse_loop
        running.clear()

        ratio_plot = np.mean(ratio_plot, axis=0)

        if ratios_pulse:
            ratio_val = (np.median if RATIO_METHOD=='median' else np.mean)(ratios_pulse)
        else:
            ratio_val = np.nan+1j*np.nan

        mag_db = 20*np.log10(np.abs(ratio_val+1e-12))
        phase_deg = np.angle(ratio_val, deg=True)

        print(f"Mag_db = {mag_db:.2f}, phase_deg = {phase_deg:.2f}")

        mag_plot = 20*np.log10(np.abs(ratio_plot)+1e-12)
        phase_deg_plot = np.angle(ratio_plot, deg=True)
        freq_plot = np.fft.fftfreq(num_samples_tx, d=1/sample_rate) / 1e6

    except Exception as e:
        print(f"Erro inesperado na configuração da recepção: {e}")
    finally:
        print("Parando recepção Pulse")


# Criar uma instância do dispositivo bladeRF
try:
    sdr = _bladerf.BladeRF()
except _bladerf.BladeRFError as e:
    print(f"Erro ao abrir o dispositivo bladeRF: {e}")
    sys.exit(1)

# Select Channels
tx_pulse_ch = _bladerf.CHANNEL_TX(0) # Pulse Drive (Pulse Protocols)
tx_ress_ch  = _bladerf.CHANNEL_TX(1) # Pulse Ressonator (Pulse Protocols)

tx     = _bladerf.CHANNEL_TX(1) # TX Tone Channel (SWEEP)
rx_ref = _bladerf.CHANNEL_RX(0) # Reference
rx_meas = _bladerf.CHANNEL_RX(1) # Measure

running.set()

# Create Threads
thread_sweep_tx = threading.Thread(target=tx_sweep, args=(sdr, tx))
thread_sweep_rx = threading.Thread(target=rx_sweep, args=(sdr, tx, rx_ref, rx_meas))

thread_pulse_tx = threading.Thread(target=tx_pulse, args=(sdr, tx_pulse_ch, tx_ress_ch))
thread_pulse_rx = threading.Thread(target=rx_pulse, args=(sdr, rx_ref, rx_meas))


# Start Sweep
thread_sweep_tx.start()
time.sleep(0.05)
thread_sweep_rx.start()

thread_sweep_tx.join()
thread_sweep_rx.join()

# print("Restarting Blade...")
# sdr.close()
# time.sleep(3)

# try:
#     sdr = _bladerf.BladeRF()
# except _bladerf.BladeRFError as e:
#     print(f"Erro ao abrir o dispositivo bladeRF: {e}")
#     sys.exit(1)


print("Starting Pulse Protocol...")



# Select the frequency for the Pulse Protocol
subset_freq = np.array([float(i[0]) for i in s21_list])
subset_mag = np.array([float(i[1]) for i in s21_list])

max_mag = subset_mag.min()

index_max_mag = np.where(subset_mag == max_mag)[0][0]
max_mag_frequency = subset_freq[index_max_mag]

print(f"Max mag frequency: {max_mag_frequency:.6}")
print(f"Max mag: {max_mag:.2f}\n")

pulse_center_freq = max_mag_frequency

running.set()
# Start Pulse Protocol
thread_pulse_tx.start()
thread_pulse_rx.start()


thread_pulse_tx.join()
thread_pulse_rx.join()

# Criar os subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

fig.suptitle(f"S21 {pulse_center_freq/1e6} MHz")

# Plot da Magnitude (ax1)
ax1.plot(np.fft.fftshift(freq_plot), np.fft.fftshift(mag_plot))
ax1.set_ylabel("Magnitude (dB)")
ax1.grid(True)


# Plot da Fase (ax2)
ax2.plot(np.fft.fftshift(freq_plot), np.fft.fftshift(phase_deg_plot))
ax2.set_ylabel("Fase (Graus)")
ax2.set_xlabel(f"Frequência (MHz) Relativo a {pulse_center_freq/1e6} MHz")
ax2.grid(True)

sdr.close()  # Garantir o fechamento do dispositivo
print("Dispositivo BladeRF fechado com sucesso")
