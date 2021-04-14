import numpy as np
from scipy.fft import fft, ifft, fftfreq


def generate_sine_wave(freq, sample_rate, duration):
    """
       Outputs a sine wave given, freq, samp_rate,
       and duration

       Args
       ----
       freq
       sample_rate
       duration

       Return
       ------
       t = time arr
       signal = sine amplitude arr

    """
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    freq_arr = t * freq
    # 2pi because np.sin takes radians
    signal = np.sin((2 * np.pi) * freq_arr)
    return t, signal


def one_sided_fft(time_arr, signal):
    """
        Takes the 1-Sided fft of a signal given the
        time series vector and signal

        Args
        ----
        Ts = 1/SAMPLE_RATE
        signal = time-series amplitude

        return
        ------
        freqs = 1-sided freq vector
        signal_fft = FT of the signal (complex form)

    """
    # Frequency vector
    Ts = time_arr[1]-time_arr[0]
    n = len(signal)
    Fs = 1 / Ts
    vec = np.linspace(0, int(n / 2), int(n / 2))  # (start: stop: len_vec)
    freq_1S = Fs * vec / n

    # Take FFT
    signal_fft_2S = fft(signal) / n  # 2-sided FFT (normalized by the len)
    signal_fft = signal_fft_2S[0:int(n / 2)]  # 1-sided FFT
    return freq_1S, signal_fft


def two_sided_fft(time_arr, signal):
    """
        Takes the 2-Sided fft of a signal given the
        time series vector and signal

        Args
        ----
        Ts = 1/SAMPLE_RATE
        signal = time-series amplitude

        return
        ------
        freqs = 1-sided freq vector
        signal_fft_2sided = FT of the signal (complex form)
   """

    # Frequency vector
    Ts = time_arr[1] - time_arr[0]
    n = len(signal)
    Fs = 1 / Ts
    freq_2S = fftfreq(n, 1 / Fs)

    # Take FFT
    signal_fft_2S = fft(signal) / n  # 2-sided FFT (normalized by the len)
    return freq_2S, signal_fft_2S


if __name__ == '__main__':
    main()
