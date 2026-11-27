import numpy as np
import pytest

from psi_lab.senales import cuadrada, espectro, senoidal


def test_numero_de_muestras():
    t, x = senoidal(f=5, fs=100, duracion=2)
    assert t.size == x.size == 200
@pytest.mark.parametrize("f", [10.0, 50.0, 120.0])
def test_pico_en_la_frecuencia_correcta(f):
    _, x = senoidal(f=f, fs=1000, duracion=1)
    freqs, amp = espectro(x, fs=1000)
    assert freqs[np.argmax(amp)] == pytest.approx(f)
def test_amplitud_recuperada():
    _, x = senoidal(f=50, fs=1000, duracion=1, amplitud=3.0)
    _, amp = espectro(x, fs=1000)
    assert amp.max() == pytest.approx(3.0, rel=1e-6)

    
def test_armonicos_impares_onda_cuadrada():
    fs = 10000
    f = 100

    _, x = cuadrada(f=f, fs=fs, duracion=1)

    freqs, amp = espectro(x, fs)

    # Armónicos impares: 100, 300 y 500 Hz
    for armonico in [1, 3, 5]:
        indice = np.argmin(np.abs(freqs - armonico * f))
        assert amp[indice] > 0.1

    # Armónicos pares: 200, 400 y 600 Hz
    for armonico in [2, 4, 6]:
        indice = np.argmin(np.abs(freqs - armonico * f))
        assert amp[indice] < 0.1