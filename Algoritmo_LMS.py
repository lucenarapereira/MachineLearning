#Precisa instalar "numpy matplotlib"

import numpy as np
import matplotlib.pyplot as plt

def algoritmo_LMS(desired, input_signal, mu, filter_order):
    """
    Parâmetros:
    - desired: Sinal desejado (target).
    - input_signal: Sinal de entrada.
    - mu: Taxa de aprendizagem.
    - filter_order: Ordem do filtro.

    Retornos:
    - output_signal: Sinal de saída do filtro.
    - error_signal: Sinal de erro.
    """
    n_samples = len(input_signal)
    output_signal = np.zeros(n_samples)
    error_signal = np.zeros(n_samples)
    weights = np.zeros(filter_order)

    for n in range(filter_order, n_samples):
        x_n = input_signal[n:n-filter_order:-1]  # Entradas passadas
        output_signal[n] = np.dot(weights, x_n)  # Saída do filtro
        error_signal[n] = desired[n] - output_signal[n]  # Sinal de erro
        weights += mu * error_signal[n] * x_n  # Atualização dos pesos

    return output_signal, error_signal

# Exemplo de uso do filtro LMS
if __name__ == "__main__":
    # Parâmetros
    n_samples = 1000
    mu = 0.01  # Taxa de aprendizagem
    filter_order = 32  # Ordem do filtro

    # Geração de um sinal de entrada e um sinal desejado
    np.random.seed(0)
    input_signal = np.random.randn(n_samples)  # Sinal de entrada (ruído branco)
    desired = np.convolve(input_signal, np.ones(filter_order)/filter_order, mode='same')  # Sinal desejado

    # Aplicar o filtro LMS
    output_signal, error_signal = algoritmo_LMS(desired, input_signal, mu, filter_order)

    # Plotar resultados
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title("Sinal de Entrada")
    plt.plot(input_signal, label="Sinal de Entrada")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("Sinal Desejado")
    plt.plot(desired, label="Sinal Desejado", color='orange')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("Sinal de Saída do Filtro LMS")
    plt.plot(output_signal, label="Sinal de Saída", color='green')
    plt.legend()

    plt.tight_layout()
    plt.show()
