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

from bladerf import _bladerf     # binding oficial da bladeRF

# ──────────────────────────────────────────────────────────────
#  ▒▒  Argumentos de linha de comando
# ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()

# --- S21 & aquisição
ap.add_argument('--window',  choices=['hanning', 'bharris'], default='bharris', help='janela FFT p/ S21 (Blackman-Harris ou Hanning)')
ap.add_argument('--blocks',  type=int, default=64, help='blocos válidos agregados por frequência')
ap.add_argument('--samples', type=int, default=2048, help='amostras por bloco/canal (FFT_SIZE do monitor)')
ap.add_argument('--method',  choices=['mean', 'median'], default='median', help='agregação (median é mais robusto a outlier)')
ap.add_argument('--dwell',   type=float, default=0.1, help='tempo de settling após set_frequency (s)')
ap.add_argument('--throw',   type=int,   default=3, help='blocos descartados logo após mudar freq')
ap.add_argument('--min_ref_bin', type=float, default=1e-9, help='|Fref[k]| mínimo para aceitar bloco (evita div/0)')

# --- RSSI/saturação
ap.add_argument('--rssi_thresh_dbfs', type=float, default=-6.0, help='RSSI_ref acima → marca SATURACAO')

# --- Saída e logging
ap.add_argument('--output', default='s21.s2p', help='base do nome de saída')
ap.add_argument('--log',    default='WARNING')

# --- Servo Auto-TX
ap.add_argument('--auto_tx',      action='store_true', help='liga servo para manter RSSI_ref≈target_dbfs')
ap.add_argument('--target_dbfs',  type=float, default=-10.0, help='alvo RMS (dBFS) do canal de referência')
ap.add_argument('--tol_db',       type=float, default=0.1, help='tolerância do servo (±dB)')
ap.add_argument('--tx_step',      type=float, default=2, help='passo de ajuste do ganho TX (dB, >=1)')
ap.add_argument('--max_iter',     type=int,   default=10, help='máx iterações do servo por frequência')
ap.add_argument('--tx_gain_min',  type=int,   default=-15, help='limite inferior do gain TX (bladeRF)')
ap.add_argument('--tx_gain_max',  type=int,   default=60, help='limite superior do gain TX (bladeRF)')

# --- LUT
ap.add_argument('--txlut', type=str, default=None, help='CSV freq,tx_gain_dB para aplicar ganho pré-calibrado')

args = ap.parse_args()
logging.basicConfig(level=getattr(logging, args.log.upper()))

# ──────────────────────────────────────────────────────────────
#  ▒▒  Parâmetros “fixos” da demo
# ──────────────────────────────────────────────────────────────
Fs          = 5e6                # taxa de amostragem (3 MS/s)
NSAMP       = args.samples       # pontos por bloco
NBLK        = args.blocks
N_THROW     = args.throw
F_TONE      = 40e3               # tom transmitido (KHz)
GAIN_TX0    = 40                 # ganho inicial de TX
GAIN_RX0    = 35                 # ganho RX0
GAIN_RX1    = 35                 # ganho RX1
FSTART      = 400e6               # start do sweep
FSTOP       = 3.5e9                # fim do sweep
FSTEP       = 10e6               # step do sweep
S2P_FILE    = args.output
running     = threading.Event()

txlut = []             #  ← declare aqui!
results = []           #  (freq, |S21|, fase, rssi, sat, g_tx)

# ──────────────────────────────────────────────────────────────
#  ▒▒  Look-Up-Table (caso --txlut)
# ──────────────────────────────────────────────────────────────
if args.txlut:
    if not os.path.isfile(args.txlut):
        print(f'Erro: arquivo {args.txlut} não encontrado', file=sys.stderr)
        sys.exit(1)
    lut_freq, lut_gain = np.loadtxt(args.txlut, delimiter=',', unpack=True, skiprows=1)
    idx_sort = np.argsort(lut_freq)
    lut_freq, lut_gain = lut_freq[idx_sort], lut_gain[idx_sort]
else:
    lut_freq = lut_gain = None  # → sem LUT

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
W = blackman_harris(NSAMP) if args.window == 'bharris' else np.hanning(NSAMP)

def s21_bins(ref_iq, meas_iq, k):
    """↦ bin k da FFT de cada canal"""
    return np.fft.fft(ref_iq)[k], np.fft.fft(meas_iq)[k]

def measure_rssi_quick(sdr, rx_ref, buf, n_call, n_avg=4):
    """medição rápida do RSSI_ref (para o servo)"""
    vals=[]
    for _ in range(n_avg):
        sdr.sync_rx(buf, n_call)
        d=np.frombuffer(buf,np.int16)
        vals.append(rssi_block_dbfs(d[0::4], d[1::4]))
    return float(np.mean(vals))

def auto_adjust_tx_gain(sdr, tx_ch, rx_ref, buf, n_call, gain_cur):
    """Servo PID simplificado: ±tx_step até |err|<=tol_db."""
    if not args.auto_tx:  # servo desligado → só mede
        return gain_cur, measure_rssi_quick(sdr, rx_ref, buf, n_call, 4)

    g = gain_cur
    for _ in range(args.max_iter):
        rssi = measure_rssi_quick(sdr, rx_ref, buf, n_call, 4)
        err  = args.target_dbfs - rssi
        if abs(err) <= args.tol_db:
            return g, rssi
        step = args.tx_step if err > 0 else -args.tx_step
        g_new = int(round(g + step))
        g_new = max(args.tx_gain_min, min(args.tx_gain_max, g_new))
        if g_new == g:   # já saturou nos limites
            return g, rssi
        sdr.set_gain(tx_ch, g_new)
        g = g_new
        time.sleep(0.01)
        for _ in range(2): sdr.sync_rx(buf, n_call)
    # retorna o último valor (não convergiu 100%)
    return g, measure_rssi_quick(sdr, rx_ref, buf, n_call, 4)

# ──────────────────────────────────────────────────────────────
#  ▒▒  Cabeçalho S2P
# ──────────────────────────────────────────────────────────────
with open(S2P_FILE, 'w') as f:
    f.write(f"! S21 sweep bladeRF  {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"! Window={args.window}  Blocks={NBLK}  Samples={NSAMP}  Method={args.method}\n")
    if lut_gain is not None:
        f.write(f"! TX LUT = {args.txlut}\n")
    else:
        f.write(f"! AutoTX={'ON' if args.auto_tx else 'OFF'}  "
                f"Target={args.target_dbfs} tol={args.tol_db}dB step={args.tx_step}dB\n")
    f.write("# Hz S DB R 50\n")

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
        while running.is_set():
            sdr.sync_tx(payload, NSAMP)
    finally:
        sdr.enable_module(tx_ch,False)

# ──────────────────────────────────────────────────────────────
#  ▒▒  Thread RX: faz o sweep de frequência
# ──────────────────────────────────────────────────────────────
def rx_loop(sdr, tx_ch, rx_ref, rx_meas):
    global txlut

    # ⚑ 1. Configura ambos RX em modo Manual
    for ch,g in ((rx_ref,GAIN_RX0),(rx_meas,GAIN_RX1)):
        sdr.set_sample_rate(ch,int(Fs))
        sdr.set_bandwidth(ch,int(Fs/2))
        sdr.set_gain(ch,g)
        sdr.set_gain_mode(ch,_bladerf.GainMode.Manual)

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
         open(S2P_FILE.replace('.s2p','_rssi_ref.csv'),'w') as fpow,\
         (open(S2P_FILE.replace('.s2p','_txlut.csv'),'w')
          if args.auto_tx and lut_gain is None else open(os.devnull,'w')) as flut,\
         tqdm(freqs,ncols=100,
              desc=('S21 sweep (TXLUT)' if lut_gain is not None else
                    ('S21 sweep (AUTO-TX)' if args.auto_tx else 'S21 sweep'))) as bar:

        # cabeçalho CSV
        fpow.write("freq_Hz,rssi_ref_dBFS,sat,tx_gain_dB\n")
        if args.auto_tx and lut_gain is None:
            flut.write("freq_Hz,tx_gain_dB\n")

        # ⚑ 3. Loop principal sobre as frequências
        for f_rf in bar:
            # 3a. sintoniza LO único (TX + RX0 + RX1)
            for ch in (tx_ch, rx_ref, rx_meas):
                sdr.set_frequency(ch, int(f_rf))
            time.sleep(args.dwell)

            # 3b. descarta blocos iniciais (N_THROW) para purgar transiente
            for _ in range(N_THROW):
                sdr.sync_rx(buf_rx, N_CALL)

            # 3c. SE LUT existe → aplica; SENÃO → chama servo (opcional)
            if lut_gain is not None:
                idx = np.searchsorted(lut_freq, f_rf, side='right') - 1
                idx = np.clip(idx, 0, len(lut_gain)-1)
                g_tx = int(lut_gain[idx])
                sdr.set_gain(tx_ch, g_tx)
                quick_rssi = measure_rssi_quick(sdr, rx_ref, buf_rx, N_CALL, 2)
            else:
                g_tx, quick_rssi = auto_adjust_tx_gain(
                        sdr, tx_ch, rx_ref, buf_rx, N_CALL, g_tx)
                if args.auto_tx:          # grava LUT “on the fly”
                    txlut.append((f_rf, g_tx))
                    flut.write(f"{f_rf:.0f},{g_tx}\n")

            # 3d. descarta mais dois blocos após mudar TxGain
            for _ in range(2): sdr.sync_rx(buf_rx,N_CALL)

            # ⚑ 4. Coleta NBLK blocos “válidos”
            rssi_blk = []
            ratios   = []
            for _ in range(NBLK):
                sdr.sync_rx(buf_rx, N_CALL)
                d = np.frombuffer(buf_rx, np.int16)

                ir, qr = d[0::4], d[1::4]         # RX0
                im, qm = d[2::4], d[3::4]         # RX1

                rssi_blk.append(rssi_block_dbfs(ir, qr))

                ref_iq  = (ir+1j*qr) * W
                meas_iq = (im+1j*qm) * W
                Fref_k, Fmeas_k = s21_bins(ref_iq, meas_iq, k_bin)
                if abs(Fref_k) > args.min_ref_bin:
                    ratios.append(Fmeas_k / Fref_k)

            # 4a. Agrega RSSI_ref
            rssi_ref = (np.median if args.method=='median' else np.mean)(rssi_blk)
            sat = rssi_ref > args.rssi_thresh_dbfs

            # 4b. Agrega S21
            if ratios:
                ratio_val = (np.median if args.method=='median' else np.mean)(ratios)
            else:
                ratio_val = np.nan+1j*np.nan
            mag_db = 20*np.log10(abs(ratio_val)+1e-12)
            phase_deg = np.angle(ratio_val, deg=True)

            # 4c. Console + arquivos
            bar.write(f"{f_rf/1e6:7.2f} MHz |S21|={mag_db:6.2f} dB  "
                      f"RSSI={rssi_ref:6.2f} dBFS {'SAT' if sat else ''}  "
                      f"TX={g_tx} dB")
            f_s2p.write(f"{f_rf:.6e} {mag_db:.6f} {phase_deg:.6f}\n")
            fpow.write(f"{f_rf:.0f},{rssi_ref:.2f},{int(sat)},{g_tx}\n")
            results.append((f_rf, mag_db, phase_deg, rssi_ref, sat, g_tx))

    running.clear()

# ──────────────────────────────────────────────────────────────
#  ▒▒  MAIN – Lança threads
# ──────────────────────────────────────────────────────────────
running.set()
sdr = _bladerf.BladeRF()
tx     = _bladerf.CHANNEL_TX(0)
rx_ref = _bladerf.CHANNEL_RX(0)
rx_mea = _bladerf.CHANNEL_RX(1)

thr_tx = threading.Thread(target=tx_loop, args=(sdr, tx))
thr_rx = threading.Thread(target=rx_loop, args=(sdr, tx, rx_ref, rx_mea))
thr_tx.start(); time.sleep(0.05); thr_rx.start()
thr_rx.join(); thr_tx.join(); sdr.close()

# ──────────────────────────────────────────────────────────────
#  ▒▒  Plot opcional
# ──────────────────────────────────────────────────────────────
if results:
    fGHz = np.array([r[0] for r in results]) / 1e9
    mag  = np.array([r[1] for r in results])
    ph   = np.array([r[2] for r in results])
    rssi = np.array([r[3] for r in results])
    sat  = np.array([r[4] for r in results], bool)
    g_tx = np.array([r[5] for r in results])

    ph_unw = np.rad2deg(np.unwrap(np.deg2rad(ph))); ph_unw -= ph_unw[0]

    plt.figure(figsize=(12,14))
    plt.subplot(411); plt.plot(fGHz, mag); plt.ylabel('|S21| (dB)')
    plt.title('S21, RSSI_ref e TX gain'); plt.grid()
    plt.subplot(412); plt.plot(fGHz, ph_unw); plt.ylabel('Fase (°)'); plt.grid()
    plt.subplot(413)
    plt.plot(fGHz[~sat], rssi[~sat], '.', label='OK')
    if sat.any(): plt.plot(fGHz[sat], rssi[sat], 'r.', label='SAT')
    plt.axhline(args.rssi_thresh_dbfs, color='g', ls='--', lw=0.8, label='limiar')
    plt.ylabel('RSSI_ref (dBFS)'); plt.grid(); plt.legend()
    plt.subplot(414); plt.plot(fGHz, g_tx, '-'); plt.xlabel('Freq (GHz)')
    plt.ylabel('TX gain (dB)'); plt.grid()
    plt.tight_layout(); plt.show()
else:
    print("Nenhum dado capturado.")
