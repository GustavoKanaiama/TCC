import numpy as np
import sys
from math import floor
from matplotlib import pyplot as plt


def moving_mean_numpy(data, window_size):

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    
    arr = np.asarray(data, dtype=float)
    if arr.size < window_size:
        raise ValueError("window_size cannot be larger than data length.")
    
    ratio = np.ones(window_size) / window_size
    return np.convolve(arr, ratio, mode='valid')


def read_s2p(filename):

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

    freq_ghz = np.array(frequencies) / 1e9
    mag_db = np.array(magnitudes_db)
    mag_meas_db = np.array(magnitude_meas_db)
    phase_deg = np.array(phases_deg)


    return freq_ghz, mag_db, mag_meas_db, phase_deg


def eval_data(foldername, num_files, window_moving_mean, onefile_only=False):
    freq = []
    mag_db = []
    mag_meas_db = []
    phase_deg = []

    if onefile_only:
        freq, mag_db, mag_meas_db, phase_deg = read_s2p(foldername)
        num_files=1

        mean_mag_db = mag_db
        mean_mag_meas_db = mag_meas_db

    else:

        for i in range(num_files):

            filename = f"{foldername}/s21_{i}.s2p"

            freq, mag_db, mag_meas_db, phase_deg = read_s2p(filename)

            if i == 0:
                mean_mag_db = np.zeros(len(mag_db))
                mean_mag_meas_db = np.zeros(len(mag_meas_db))

            #print(mag_meas_db)

            mean_mag_db += mag_db
            mean_mag_meas_db += mag_meas_db

    mean_mag_db /= num_files
    mean_mag_meas_db /= num_files

    result_mean = np.pad(moving_mean_numpy(mean_mag_meas_db, window_moving_mean), (floor(window_moving_mean/2), floor(window_moving_mean/2)), mode='edge')
    mean_mag_meas_db = result_mean

    return freq, mean_mag_db, mean_mag_meas_db

def plot_data(freq_axis, mean_mag_db_axis, mean_mag_meas_db_axis):
    # Cria uma figura com dois subplots (um em cima do outro, compartilhando o eixo X)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    fig.suptitle(f'S21 - Rel. Meas.', fontsize=16)
    top_ylim=-15
    bottom_ylim=-70

    # --- Plot 1: Magnitude Relativa ---
    ax1.plot(freq_axis, mean_mag_db_axis, 'b.-', label='S21 Relativo')
    ax1.set_ylabel('Magnitude "Relativa" (dB)')
    ax1.grid(True, which='both', linestyle='--')
    ax1.set_xlim(5.08, 5.72)
    ax1.set_ylim(bottom_ylim, top_ylim)
    ax1.legend()

    # --- Plot 2: Magnitude Absoluta (meas) ---
    ax2.plot(freq_axis, mean_mag_meas_db_axis, 'r.-', label='S21 Meas')
    ax2.set_ylabel('Magnitude "Meas" (dB)')
    ax2.set_xlabel('Frequência (GHz)')
    ax2.set_ylim(bottom_ylim, top_ylim)
    
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend()
    # Ajusta o layout para evitar sobreposição de títulos
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()

# num_files = 11
# window_moving_mean = 5

# foldername = "./sweep/Save_s21/Batch_01/"

# freq, mean_db, meas_db = eval_data(foldername, num_files, window_moving_mean)

# plot_data(freq, mean_db, meas_db)