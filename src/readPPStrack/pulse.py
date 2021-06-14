#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:30:14 2021

@author: lutzray
"""

import matplotlib.pyplot as plt
import numpy as np
from sys import exit
from pydub import AudioSegment, silence
from numpy import sin, pi, exp
from lmfit import Parameters, fit_report, minimize
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import hilbert
import scipy.signal


# filename = '../2SinBipNikonFar.wav'
# filename = '../2SinBipNikon.wav' # all clipped
# filename = '/Volumes/secondaire/Nextcloud/Sync/Dev/AtomicSync/PostProduction/NikClics.wav'
# filename = '/Volumes/secondaire/Nextcloud/Sync/Dev/AtomicSync/PostProduction/CanonClics.wav'

filename =  '../../../CanonClics.wav'

import numpy as np
import soundfile as sf



def make_seconds(framerate):
    def f(sample):
        # return the time for sample index
        return sample/framerate
    return f

def make_samples(framerate):
    def f(seconds):
        # return the index for seconds
        return seconds*framerate
    return f




# wholeFileAudioSegment = AudioSegment.from_wav(filename)
# points = wholeFileAudioSegment.get_array_of_samples()

# # signal_data = signal_data[:,1][20000:]
# signal_data = signal_data[:,1]

# time = make_seconds(samplerate)
# index = make_samples(samplerate)


# # rms = [np.sqrt(np.mean(block**2)) for block in
# rms = [np.abs(block).max() for block in
#        sf.blocks(filename, blocksize=1024, overlap=512)]


# steps_blck = int(len(signal_data)/len(rms))
# x_blocks = range(0,len(signal_data), steps_blck) 




def findSinusFittingWordsAndSilences(signal_data):
    analytic_signal = hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    x = np.arange(len(envelope))
    phaseParam = Parameters()
    phaseParam.add('phase', value=0)
    def sinusFittingPhaseOnly(pars, x, signal_data=None):
        # looking for phase sx with a sin of 1 sec period and 0<y<1.0
        vals = pars.valuesdict()
        A = 1
        p = samplerate
        phi = vals['phase']
        y0 = 1
        model = y0 + A*sin(x*2*pi/p + phi)
        if signal_data is None:
            return model
        return model - signal_data
    fitPhase = minimize(sinusFittingPhaseOnly, phaseParam, args=(x,) , kws={'signal_data': envelope})
    # print(fit_report(out))
    AsinwtPlusPhiParams = Parameters()
    AsinwtPlusPhiParams.add('A', value=0.25, min=0, max=1)
    AsinwtPlusPhiParams.add('period', value=samplerate)
    AsinwtPlusPhiParams.add('phase', value=fitPhase.params.valuesdict()['phase']) # phi from first fit
    AsinwtPlusPhiParams.add('y0', value=0.25, min=0, max=0.5)
    def sinusFittingAll(pars, x, signal_data=None):
        vals = pars.valuesdict()
        A = vals['A']
        p = vals['period']
        phi = vals['phase']
        y0 = vals['y0']
        model = y0 + A*sin(x*2*pi/p + phi)
        if signal_data is None:
            return model
        return model - signal_data
    finalFit = minimize(sinusFittingAll, AsinwtPlusPhiParams, args=(x,),
                        kws={'signal_data': envelope})
    # trying with phase found above
    fittedValues = sinusFittingAll(finalFit.params, x)
    return (finalFit.params, fittedValues)

def find_PPS_near_word_start(
        signal_data,
        word_start,
        silence_interval_ms=(-15,-4),
        max_pulse_distance=3):
    """
    Locates the rise of PPS square pulse in audio signal_data looking for the 
    first time the level exceed the maximum value found in the specified 
    interval silence. Search is done on the absolute of the signal to catch 
    negative edges too. 
    

    Parameters
    ----------
    signal_data : 1D numpy.ndarray of [-1.0, +1.0] floats 
        audio data
    word_start : integer
        indice of the begining of modulated word
    silence_interval_ms : tuple of 2 integer, optional
        default is (-15,-4) in milliseonds.
        Where to evaluate silence levels; indices are relative to word_start.
        The search begins at the silence interval end,
        ie at indice = word_start + silence_interval_ms[1]
    max_pulse_distance : integer, default is 3 ms.
        how far relative to word_start the pulse can be: if too far log a 
        warming.


    Returns
    -------
    An integer, position of the detected PPS sync pulse
    flag if the search has been successfull
    """

    """For an illustration paste these lines in a terminal (including EOF):
    cat << EOF | base64 -D | gzip -d > clics.find_PPS_near_word_start.svg
H4sICDBaxmAAA2ZpbmRfUFBTX25lYXJfd29yZF9zdGFydC5zdmcA5Vdbb9NIFH7vr5idfQEpHs/NNxQ
XCZcWpACVKLtaIRRZjZuMSOzIHpLCr+fMeOx6N0kpDdI+YFmOv/GZ73znkuNk/Px2tUSbom5UVaaYEY
pRUV5XM1XOU/zh6tyL8fPTk/EfZ++yq38uX6JmM0eXH15MXmcIe77/t8h8/+zqDL3/6wIxwnz/5VuM8
ELr9TPf3263ZCtIVc/9izpfL9R144Ohbwxhkw9kjJGZnmFwYZgHQhhGIK1s0j1knFJqNjuTZ7dLVX7e
Z8iSJPHtUzBNMQT31V63aqYXKY7CGKNFoeYLDS4plxhtVLF9URnbEYITLEbIPjk9QWg8R2qW4kn+tai
nzC7B4jrXCwTLbzgLSTCSgpMATUJKO4DRjVouvWqdXyvdCmh0XX0uUvwntUe34DlhDPuOfN5+Dt3IMD
bMNDZugijsQRbEgjAeORjE1o4JEkeiQ5zaXS2Qst1lQcDMXtRzBKHhaOnt/cQ5tiCTITO8Dtprz+GQp
Z84YB1nDjhNPYfVO4zrW5szSNC5Pbp8/KZ5eFzv+PN7mojf5WsyBJnkCYmZ7MSYBgahEFAc9qgNqAVB
G5ADwuxFPUebFD5M3gCAL2F4ezv6Lw7a008csI6zDjhNHUerdxDKg5ro98nDkU2ki1uNdJ2XzU1Vr1K
8ynWtbp8wOyfhhJsApAQ0Gpn8PMV9snWzzkszfz2WmOd2Cgd2Llal9m7ylVrCyqtiuSm0us7deqO+gU
AW91V0Uk+3VT0b+5a1E2nE3a8TFHrMqqRGX0AiGY5smpIk3C8W3oY0sGI9acwfppft6L28fH9Y7l0n7
g6LgMr73x7nUPTd4sm+eD25EI4cTdrbX0fNedxRt7dHUx+qIElCKBklPEigmO7TrQrKiIhNDwpJWBLt
rWjEScxZW1FGov82YFZ9qVVR/6icjVrCT6Riqkpd1Jt8OV01H+mnn2/I+8KRNPk/w2G/OJyAR0RwYcI
RJJT7v28RJWFyfDTll2ba6LzWPx3CcJQJFkEEZkDEe8VKBo+tWP5YofMSkj3LdX6cUMnhEh2YtzxsE8
oflE6+o/JGlbMpzK9pWeT11AzePbkdupQh9GR8nNMnB9nvArIZfQQ30A0yP/qxJ3GEn74VH+CHRUc42
vkGpx89Fow8+enpgeay7/Wx+RN1evIdpKaqbgUOAAA=
EOF
    """
    max_silence_level = np.abs(signal_data).max()
    return

signal_data, samplerate = sf.read(filename)
sec = make_seconds(samplerate)
sample = make_samples(samplerate)
finalFitParams, fitted_y = findSinusFittingWordsAndSilences(signal_data)
period = finalFitParams.valuesdict()['period']
phase = finalFitParams.valuesdict()['phase']
i0 = -phase*period/(2*pi)
print('fit freq/framerate: %f,  t0: %i samples %f sec, %i'%
      (period/samplerate, int(i0), sec(i0), period))
x = np.arange(len(signal_data))
fig, ax = plt.subplots()
ax.plot(fitted_y - finalFitParams['y0'],  marker='.', markersize='0.1',  linestyle="None", color='orangered')
ax.plot(signal_data,  marker='.', markersize='0.3', linestyle="None", color='black')
ax.hlines(y=0, xmin=0, xmax=len(x), linewidth=1, color='black')

plt.show()