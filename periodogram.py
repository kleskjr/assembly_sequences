import numpy
from scipy import fft

def periodogram(x, shift, nfft, ts=1):
   def sliding_window(x):
       n = len(x)
       byte = x.nbytes/n
       max_shift = n-nfft
       assert max_shift > 0
       n_frames = int(max_shift/shift)+1
       x_slide = numpy.lib.stride_tricks.as_strided(x, shape=(nfft, n_frames),
                                                strides=(byte, shift*byte))
       return x_slide

   x_windows = sliding_window(x)
   psd = numpy.abs(fft(x_windows,nfft,axis=0))
   psd = psd.mean(1)
   f = numpy.arange(nfft/2+1)/(nfft*ts)
   return f, psd[:nfft/2+1]

