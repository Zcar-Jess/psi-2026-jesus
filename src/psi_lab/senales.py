"""Generacion de senales muestreadas y espectro de amplitud."""
import numpy as np


def senoidal(f,fs,duracion,amplitud=1.0, fase=0.0):
    n = round(fs * duracion)  # Numero de muestras
    t = np.arange(n) / fs  # Instantes de muestreo
    return t, amplitud * np.sin(2 * np.pi * f * t + fase)

def espectro(x, fs):
    x = np.asarray(x, dtype=float)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amp = np.abs(np.fft.rfft(x)) / n

    amp[1 : (n + 1) // 2] *= 2

    return freqs, amp

def cuadrada(f, fs, duracion):
    t, x = senoidal(f, fs, duracion)
    return t, np.sign(x)