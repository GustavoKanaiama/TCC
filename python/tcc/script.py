import numpy as np
import sys

S2P_FILE    = './sweep/s21.s2p'

# Listas para armazenar os dados
frequencies = []
magnitudes_db = []
phases_deg = []

print(f"Lendo o arquivo: {S2P_FILE}")

try:
    with open(S2P_FILE, 'r') as f:
        for line in f:
            # Remove espaços em branco no início/fim
            line = line.strip()
            
            # Ignora linhas vazias
            if not line:
                continue
            
            # Ignora linhas de comentário (começam com '!')
            if line.startswith('!'):
                continue
            
            # Ignora a linha de cabeçalho (começa com '#' header)
            if line.startswith('#'):
                continue
            
            # Processa a linha de dados
            try:
                parts = line.split()
                if len(parts) >= 3:
                    frequencies.append(float(parts[0]))
                    magnitudes_db.append(float(parts[1]))
                    phases_deg.append(float(parts[2]))
            except ValueError as e:
                print(f"Aviso: Ignorando linha mal formatada: '{line}' ({e})")

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em '{S2P_FILE}'")
    sys.exit(1)
except Exception as e:
    print(f"Erro ao ler o arquivo: {e}")
    sys.exit(1)

if not frequencies:
    print("Erro: Nenhum dado válido encontrado no arquivo.")
    sys.exit(1)


freq_ghz = np.array(frequencies) / 1e9
mag_db = np.array(magnitudes_db)
phase_deg = np.array(phases_deg)


