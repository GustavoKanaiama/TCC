#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s21_sweep_bladerf_rssi_ref_autotx         v2025-07-29

Mede S21 (magnitude + fase) usando uma única bladeRF + divisor 3 dB.
Inclui medição de RSSI no canal de referência (RX0) em dBFS
e três modos de controle do transmissor (TX):

    1. TX FIXO      : usa GAIN_TX0 em toda a faixa (opção mais simples)
    2. AUTO-TX      : servo que ajusta gain dB para manter RSSI_ref --target_dbfs
    3. LUT de ganho : aplica a tabela gerada por um sweep anterior (--txlut)

Arquivos gerados
----------------
<output>.s2p                 | S-parâmetro S21 complexo (freq, |S21|, fase)
<output>_rssi_ref.csv        | freq_Hz, RSSI_ref_dBFS, saturacao(0/1), tx_gain_dB
<output>_txlut.csv           | freq_Hz, tx_gain_dB     (somente se --auto_tx)

Fluxo típico
------------
  # 1) Gera LUT rodando o servo
  python3 sweep_s21.py --auto_tx --target_dbfs -8 --output thru

  # 2) Mede DUT reutilizando o mesmo ganho f-a-f
  python3 sweep_s21.py --txlut thru_txlut.csv --output dut_raw

  # 3) Pós-processo: S21_corr = dut_raw / thru   (ver docstring)

"""

# ──────────────────────────────────────────────────────────────
#  ▒▒  Importações
# ──────────────────────────────────────────────────────────────
import argparse, threading, time, os, sys, logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import math

from bladerf import _bladerf     # binding oficial da bladeRF


# ──────────────────────────────────────────────────────────────
#  ▒▒  Parâmetros do Servo Auto-TX
# ──────────────────────────────────────────────────────────────
AUTO_TX = False
TARGET_DBFS = -10 # alvo RMS (dBFS) do canal de referência
TOL_DB = 0.1 # tolerância do servo (±dB)
TX_STEP = 2 # passo de ajuste do ganho TX (dB, >=1)
MAX_ITER = 10 # máx iterações do servo por frequência
TX_GAIN_MIN = 0 # limite inferior do gain TX (bladeRF)
TX_GAIN_MAX = 60 # limite superior do gain TX (bladeRF)

# ──────────────────────────────────────────────────────────────
#  ▒▒  Parâmetros “fixos” da demo
# ──────────────────────────────────────────────────────────────
Fs          = 5e6                # taxa de amostragem (3 MS/s)
NSAMP       = 2048                  # amostras por bloco/canal (FFT_SIZE do monitor)
NBLK        = 64                    # blocos válidos agregados por frequência
N_THROW     = 3                 # blocos descartados logo após mudar freq
F_TONE      = 40e4               # tom transmitido (KHz)
GAIN_TX0    = 38                 # ganho inicial de TX
GAIN_RX0    = 50                 # ganho RX0
GAIN_RX1    = 50                 # ganho RX1
FSTART      = 5.1e9              # start do sweep
FSTOP       = 5.7e9              # fim do sweep
FSTEP       = 0.25e6               # step do sweep
WINDOW      = 'bharris'         # ou 'hanning' janela FFT p/ S21 (Blackman-Harris ou Hanning)
METHOD      = 'median'            # ou 'median', agregação (median é mais robusto a outlier)
DWELL       = 0.1               # tempo de settling após set_frequency (s)
MIN_REF_BIN = 1e-9              # |Fref[k]| mínimo para aceitar bloco (evita div/0)

NUM_SWEEPS  = 1
TIMEOUT     = 5000              # em millissegundos

# Batch_02 -> apenas o fio, sem o ressonador
# Batch_03 -> com o ressonador

S2P_FILE    = './sweep/s21.s2p'
FOLDERNAME  = "./sweep/Save_s21/Batch_04/"

#COPY_FILE   = f'./sweep/Save_s21/Batch_01/s21_test.s2p'

running     = threading.Event()

txlut = []             #  ← declare aqui!
#results = []           #  (freq, |S21|, fase)

freq_plot  = mag_plot = phase_plot = []

lut_freq = lut_gain = None  # → sem LUT
flag_finish = False

# TODO: função para checar a variação do "ref", este não pode ter bruscas variações.

# ──────────────────────────────────────────────────────────────
#  ▒▒  Funções auxiliares
# ──────────────────────────────────────────────────────────────
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
W = blackman_harris(NSAMP) if WINDOW == 'bharris' else np.hanning(NSAMP)

def s21_bins(ref_iq, meas_iq, k):
    """↦ bin k da FFT de cada canal"""
    return np.fft.fft(ref_iq)[k], np.fft.fft(meas_iq)[k]


# ──────────────────────────────────────────────────────────────
#  ▒▒  Cabeçalho S2P
# ──────────────────────────────────────────────────────────────
with open(S2P_FILE, 'w') as f:
    f.write(f"! S21 sweep bladeRF - TX0={GAIN_TX0} db | RX0={GAIN_RX0} db | RX1={GAIN_RX1} db - {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"# Frequency [MHz], |S21| [db], |Mag_meas| [db], Phase [deg]\n")


def plot_s2p(filename):

    # Listas para armazenar os dados
    frequencies = []
    magnitudes_db = []
    magnitude_meas_db = []
    phases_deg = []

    print(f"Lendo o arquivo: {filename}")

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Remove espaços em branco no início/fim
                line = line.strip()
                
                # Ignora linhas vazias
                if not line:
                    continue
                
                # Ignora linhas de comentário (começam com '!')
                if line.startswith('!'):
                    continue
                
                # Ignora a linha de cabeçalho (começa com '#')
                if line.startswith('#'):
                    continue
                
                # Processa a linha de dados
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        frequencies.append(float(parts[0]))
                        magnitudes_db.append(float(parts[1]))
                        magnitude_meas_db.append(float(parts[2]))
                        phases_deg.append(float(parts[3]))
                except ValueError as e:
                    print(f"Aviso: Ignorando linha mal formatada: '{line}' ({e})")

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{filename}'")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        sys.exit(1)

    # if not frequencies:
    #     print("Erro: Nenhum dado válido encontrado no arquivo.")
    #     sys.exit(1)


    freq_ghz = np.array(frequencies) / 1e9
    mag_db = np.array(magnitudes_db)
    phase_deg = np.array(phases_deg)

    print(f"Plotando {len(freq_ghz)} pontos de dados...")

    # --- Criação do Gráfico ---
    
    # Cria uma figura com dois subplots (um em cima do outro, compartilhando o eixo X)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    fig.suptitle(f'S21 - {filename}', fontsize=16)

    # --- Plot 1: Magnitude ---
    ax1.plot(freq_ghz, mag_db, 'b.-', label='S21 Relativo')
    ax1.set_ylabel('Magnitude "Relativa" (dB)')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend()
    
    # --- Plot 2: Fase ---
    ax2.plot(freq_ghz, magnitude_meas_db, 'r.-', label='S21 Meas')
    ax2.set_ylabel('Magnitude "Meas" (dB)')
    ax2.set_xlabel('Frequência (GHz)')
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend()
    
    # Ajusta o layout para evitar sobreposição de títulos
    plt.tight_layout()
    
    # Exibe o gráfico
    plt.show()

# ──────────────────────────────────────────────────────────────
#  ▒▒  Thread TX: gera tom contínuo I/Q
# ──────────────────────────────────────────────────────────────
def tx_loop(sdr, tx_ch):
    t = np.arange(NSAMP) / Fs
    sig = 0.9 * np.exp(1j*2*np.pi*F_TONE*t)
    i16 = np.round(sig.real*2047).astype(np.int16)
    q16 = np.round(sig.imag*2047).astype(np.int16)
    inter = np.empty(NSAMP*2, np.int16); inter[0::2],inter[1::2] = i16,q16
    payload = inter.tobytes()

    sdr.set_sample_rate(tx_ch, int(Fs))
    sdr.set_bandwidth(tx_ch,  int(Fs/2))
    sdr.set_gain(tx_ch, GAIN_TX0)
    sdr.sync_config(_bladerf.ChannelLayout.TX_X1,
                    _bladerf.Format.SC16_Q11,16,NSAMP,8,3500)
    sdr.enable_module(tx_ch,True)
    try:
        while running.is_set() and not flag_finish:
            sdr.sync_tx(payload, NSAMP)
    finally:
        sdr.enable_module(tx_ch,False)
        print("Finish tx_loop")

# ──────────────────────────────────────────────────────────────
#  ▒▒  Thread RX: faz o sweep de frequência
# ──────────────────────────────────────────────────────────────
def rx_loop(sdr, tx_ch, rx_ref, rx_meas):
    global txlut, flag_finish

    for k in range(NUM_SWEEPS):
        # ⚑ 1. Configura ambos RX em modo Manual
        for ch,g in ((rx_ref,GAIN_RX0),(rx_meas,GAIN_RX1)):
            sdr.set_sample_rate(ch,int(Fs))
            sdr.set_bandwidth(ch,int(Fs/2))
            sdr.set_gain(ch,g)
            sdr.set_gain_mode(ch,_bladerf.GainMode.Manual)

        if k == 0:
            sdr.sync_config(_bladerf.ChannelLayout.RX_X2,
                            _bladerf.Format.SC16_Q11,16,NSAMP,8,3500)
            
            sdr.enable_module(rx_ref,True); sdr.enable_module(rx_meas,True)


        # ⚑ 2. Pré-aloca buffers
        freqs = np.arange(FSTART, FSTOP+FSTEP, FSTEP)
        buf_rx = bytearray(NSAMP*2*4)      # I/Q de dois canais
        N_CALL = NSAMP*2                   # nº “samples” a pedir no sync_rx
        k_bin  = int(round(F_TONE*NSAMP/Fs))

        # ganho atual de TX (parte do servo ou LUT)
        g_tx = GAIN_TX0
        if lut_gain is not None:
            g_tx = int(lut_gain[0])
            sdr.set_gain(tx_ch, g_tx)

        # Arquivos auxiliares
        with open(S2P_FILE,'a') as f_s2p,\
            tqdm(freqs,ncols=100,
                desc=('S21 sweep (TXLUT)' if lut_gain is not None else
                        (f'S21 sweep {k+1}/{NUM_SWEEPS}'))) as bar:


            # ⚑ 3. Loop principal sobre as frequências
            for f_rf in bar:
                # 3a. sintoniza LO único (TX + RX0 + RX1)
                for ch in (tx_ch, rx_ref, rx_meas):
                    sdr.set_frequency(ch, int(f_rf))
                time.sleep(DWELL)

                # 3b. descarta blocos iniciais (N_THROW) para purgar transiente
                for _ in range(N_THROW):
                    sdr.sync_rx(buf_rx, N_CALL)

                # 3d. descarta mais dois blocos após mudar TxGain
                for _ in range(2): sdr.sync_rx(buf_rx, N_CALL)

                # ⚑ 4. Coleta NBLK blocos “válidos”
                ratios   = []
                meas = []
                ref = []
                for _ in range(NBLK):
                    sdr.sync_rx(buf_rx, N_CALL)
                    d = np.frombuffer(buf_rx, np.int16)

                    ir, qr = d[0::4], d[1::4]         # RX0
                    im, qm = d[2::4], d[3::4]         # RX1

                    meas_iq  = (ir+1j*qr) #* W
                    ref_iq = (im+1j*qm) #* W
                    Fref_k, Fmeas_k = s21_bins(ref_iq, meas_iq, k_bin)
                    if abs(Fref_k) > MIN_REF_BIN:

                        ref.append(Fref_k)
                        meas.append(Fmeas_k)
                        
                        ratios.append(Fmeas_k / Fref_k)

                # 4b. Agrega S21
                if ratios:
                    ratio_val = (np.median if METHOD=='median' else np.mean)(ratios)
                    ref_val = (np.median if METHOD=='median' else np.mean)(ref)
                    meas_val = (np.median if METHOD=='median' else np.mean)(meas)
                else:
                    ratio_val = np.nan+1j*np.nan
                mag_db = 20*np.log10(abs(ratio_val)+1e-12)
                phase_deg = np.angle(ratio_val, deg=True)

                mag_db_ref = 20*np.log10(abs(ref_val)+1e-12)
                mag_db_meas = 20*np.log10(abs(meas_val)+1e-12)

                rms = np.sqrt(np.average(np.abs(ref_iq)*np.abs(ref_iq)))

                # 4c. Console + arquivos
                bar.write(f"{f_rf/1e6:7.2f} MHz |S21|={mag_db:6.2f} dB  "
                        f"RX_ref: {mag_db_ref:6.2f} | RX_meas: {mag_db_meas:6.2f} | TX={g_tx} dB | RMS={rms:.2f}")
                f_s2p.write(f"{f_rf:.6e} {mag_db:.6f} {mag_db_meas:.6f} {phase_deg:.6f}\n")


        # Copy the .s2p file to Save_s21
        copy_filename = f"{FOLDERNAME}/s21_{k}.s2p"
        shutil.copyfile(S2P_FILE, copy_filename)
        
        # Clear main .s2p file
        if k < (NUM_SWEEPS-1): 
            with open(S2P_FILE, 'w') as f: 
                f.write("")

        time.sleep(DWELL*100)

    flag_finish = True

    print("Finish RX loop")
    sdr.enable_module(_bladerf.CHANNEL_RX(0), False)
    sdr.enable_module(_bladerf.CHANNEL_RX(1), False)



# ──────────────────────────────────────────────────────────────
#  ▒▒  MAIN – Lança threads
# ──────────────────────────────────────────────────────────────
running.set()
sdr = _bladerf.BladeRF()
tx     = _bladerf.CHANNEL_TX(1)
rx_ref = _bladerf.CHANNEL_RX(0)
rx_mea = _bladerf.CHANNEL_RX(1)

thr_tx = threading.Thread(target=tx_loop, args=(sdr, tx))
thr_rx = threading.Thread(target=rx_loop, args=(sdr, tx, rx_ref, rx_mea))

thr_tx.start()
time.sleep(0.05)
thr_rx.start()

thr_rx.join()
thr_tx.join()

running.clear()

plot_s2p(S2P_FILE) # plot the last .s2p file

sdr.close()
