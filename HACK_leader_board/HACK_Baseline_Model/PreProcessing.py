import numpy as np
import pandas as pd
from obspy.signal.polarization import flinn
from scipy import signal
from obspy.signal.freqattributes import central_frequency_unwindowed


def _spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean


def preprocess(df):
    df.drop(['receiver_latitude', 'receiver_longitude', 'receiver_elevation_m', 'p_arrival_sample', 'p_travel_sec',
         's_arrival_sample', 'source_origin_time', 'source_latitude', 'source_longitude',
         'source_depth_km'], axis=1, inplace=True)
    #     Degree of rectiliniarity (polarization)
    print('     Processing - Degree of rectiliniarity (polarization)')
    df['rect_azimuth'], df['rect_incidence'], df['rect_rectilinearity'], df['rect_planarity'] = zip(*df.apply(lambda x: flinn([x['Z'],x['N'],x['E']]), axis=1))
    
    # trace-by-trace features
    print('     Starting Trace-By-Trace feature processing')
    trace_list = ['E','N','Z']
    for tl in trace_list:    
    #     SPECTRAL CENTROID
        print('          Processing Spectral Centroid')
        df['spectral_centroid_{}'.format(tl)] = df[tl].apply(_spectral_centroid)
    #     RMS of frequency amplitude
        print('          RMS of frequency amplitude')
        df['rms_freq_amp_{}'.format(tl)] = df[tl].apply( lambda x: np.sqrt(np.mean(np.square(np.real(np.fft.fft(x))))))
    #     Maximum power of frequency amplitude
        print('          Maximum power of frequency amplitude')
        df['max_power_freq_amp_{}'.format(tl)] = df[tl].apply(lambda x: np.sqrt(signal.periodogram(x, 100, 'flattop', scaling='spectrum')[1].max()))
    #     Dominant frequency
        print('          Dominant frequency')
        df['dominant_freq_{}'.format(tl)] = df[tl].apply(lambda x: central_frequency_unwindowed(x,fs=100))
        
#     return trace_id and model columns
    return df[['trace_id', 'snr_db_E', 'snr_db_N', 'snr_db_Z',
               'spectral_centroid_E', 'spectral_centroid_N', 'spectral_centroid_Z',
               'rect_azimuth', 'rect_incidence', 'rect_rectilinearity', 'rect_planarity',
               'rms_freq_amp_E', 'rms_freq_amp_N', 'rms_freq_amp_Z',
               'max_power_freq_amp_E', 'max_power_freq_amp_N', 'max_power_freq_amp_Z',
               'dominant_freq_E', 'dominant_freq_N', 'dominant_freq_Z']]