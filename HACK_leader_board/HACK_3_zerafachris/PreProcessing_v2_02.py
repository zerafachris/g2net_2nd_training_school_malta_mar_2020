import numpy as np
import pandas as pd
from obspy.signal.polarization import flinn
from scipy import signal
from obspy.signal.freqattributes import central_frequency_unwindowed
from scipy.signal import hilbert
from obspy.signal.cross_correlation import correlate as obspy_corr

def _spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean

def envelope(trace_Z):
    return np.sqrt((trace_Z**2) + (np.imag(hilbert(trace_Z))**2))

def envelope_similarity(trace_Z): 
    window_step = round(trace_Z.shape[0]/2)
    window_deep = trace_Z[window_step:]
    window_shallow = trace_Z[:window_step]
    env_sim_deep = np.sum(envelope(window_shallow)-envelope(window_deep))/envelope(window_shallow)
    env_sim_shallow = np.sum(envelope(window_deep)-envelope(window_shallow))/envelope(window_deep)
    return env_sim_deep.max(), env_sim_deep.mean(), env_sim_shallow.max(), env_sim_shallow.mean()

def waveform_xc_properties(full_signal):
    window_step = round(full_signal.shape[0]/2)
    window_deep = full_signal[window_step:]
    window_shallow = full_signal[:window_step]
    xcor_deep = obspy_corr(full_signal, window_deep, window_step)
    xcor_shallow = obspy_corr(full_signal, window_shallow, window_step)
    return xcor_deep.max(), xcor_deep.mean(), xcor_shallow.max(), xcor_shallow.mean()

def moving_average(arr, lag):
    ret = np.cumsum(arr, dtype=float)
    ret[lag:] = ret[lag:] - ret[:-lag]
    return ret[lag - 1:] / lag

def preprocess(df):
    # Rolling AVG traces
    print('Calculate rolling Avg')
    df['E_MA250'] = df['E'].apply(lambda x: moving_average(x,250))
    df['N_MA250'] = df['N'].apply(lambda x: moving_average(x,250))
    df['Z_MA250'] = df['Z'].apply(lambda x: moving_average(x,250))
    df['E_MA1000'] = df['E'].apply(lambda x: moving_average(x,1000))
    df['N_MA1000'] = df['N'].apply(lambda x: moving_average(x,1000))
    df['Z_MA1000'] = df['Z'].apply(lambda x: moving_average(x,1000))    
    
    #     Degree of rectiliniarity (polarization)
    print('     Processing - Degree of rectiliniarity (polarization)')
    df['rect_azimuth'], df['rect_incidence'], df['rect_rectilinearity'], df['rect_planarity'] = zip(*df.apply(lambda x: flinn([x['Z'],x['N'],x['E']]), axis=1))
    df['rect_azimuth_MA250'], df['rect_incidence_MA250'], df['rect_rectilinearity_MA250'],df['rect_planarity_MA250'] = zip(*df.apply(lambda x: flinn([x['Z_MA250'],x['N_MA250'],x['E_MA250']]), axis=1))
    df['rect_azimuth_MA1000'], df['rect_incidence_MA1000'], df['rect_rectilinearity_MA1000'],df['rect_planarity_MA1000'] = zip(*df.apply(lambda x: flinn([x['Z_MA1000'],x['N_MA1000'],x['E_MA1000']]), axis=1))
        
    # trace-by-trace features
    print('     Starting Trace-By-Trace feature processing')
    trace_list = ['E','N','Z']
    # Envelope similarity
    print('          Processing - Envelope Similarity')
    for tl in trace_list:
        df['{}_env_sim_deep_max'.format(tl)], df['{}_env_sim_deep_mean'.format(tl)], df['{}_env_sim_shallow_max'.format(tl)], df['{}_env_sim_shallow_mean'.format(tl)] = zip(*df[tl].apply(lambda x : envelope_similarity(x))) 

    trace_list_MA = ['E','N','Z', 'E_MA250', 'N_MA250', 'Z_MA250', 'E_MA1000', 'N_MA1000', 'Z_MA1000']
    for tl in trace_list_MA:    
    #     SPECTRAL CENTROID
        print('          Processing Spectral Centroid - {}'.format(tl))
        df['spectral_centroid_{}'.format(tl)] = df[tl].apply(_spectral_centroid)
    #     RMS of frequency amplitude
        print('          RMS of frequency amplitude - {}'.format(tl))
        df['rms_freq_amp_{}'.format(tl)] = df[tl].apply( lambda x: np.sqrt(np.mean(np.square(np.real(np.fft.fft(x))))))
    #     Maximum power of frequency amplitude
        print('          Maximum power of frequency amplitude - {}'.format(tl))
        df['max_power_freq_amp_{}'.format(tl)] = df[tl].apply(lambda x: np.sqrt(signal.periodogram(x, 100, 'flattop', scaling='spectrum')[1].max()))
    #     Dominant frequency
        print('          Dominant frequency - {}'.format(tl))
        df['dominant_freq_{}'.format(tl)] = df[tl].apply(lambda x: central_frequency_unwindowed(x,fs=100))
    #     Waveform correlation
        print('          Waveform Correlation  - {}'.format(tl))
        df['xcor_{}_deep_max'.format(tl)], df['xcor_{}_deep_mean'.format(tl)], df['xcor_{}_shallow_max'.format(tl)], df['xcor_{}_shallow_mean'.format(tl)] = zip(*df[tl].apply(lambda x : waveform_xc_properties(x)))

#     return trace_id and model columns
    return df[['trace_id','spectral_centroid_E','spectral_centroid_N','spectral_centroid_Z','spectral_centroid_E_MA250','spectral_centroid_N_MA250','spectral_centroid_Z_MA250','spectral_centroid_E_MA1000','spectral_centroid_N_MA1000','spectral_centroid_Z_MA1000','rect_azimuth','rect_incidence','rect_rectilinearity','rect_planarity','rect_azimuth_MA250','rect_incidence_MA250','rect_rectilinearity_MA250','rect_planarity_MA250','rect_azimuth_MA1000','rect_incidence_MA1000','rect_rectilinearity_MA1000','rect_planarity_MA1000','rms_freq_amp_E','rms_freq_amp_N','rms_freq_amp_Z','rms_freq_amp_E_MA250','rms_freq_amp_N_MA250','rms_freq_amp_Z_MA250','rms_freq_amp_E_MA1000','rms_freq_amp_N_MA1000','rms_freq_amp_Z_MA1000','max_power_freq_amp_E','max_power_freq_amp_N','max_power_freq_amp_Z','max_power_freq_amp_E_MA250','max_power_freq_amp_N_MA250','max_power_freq_amp_Z_MA250','max_power_freq_amp_E_MA1000','max_power_freq_amp_N_MA1000','max_power_freq_amp_Z_MA1000','dominant_freq_E','dominant_freq_N','dominant_freq_Z','dominant_freq_E_MA250','dominant_freq_N_MA250','dominant_freq_Z_MA250','dominant_freq_E_MA1000','dominant_freq_N_MA1000','dominant_freq_Z_MA1000','xcor_E_deep_max','xcor_E_deep_mean','xcor_E_shallow_max','xcor_E_shallow_mean','xcor_N_deep_max','xcor_N_deep_mean','xcor_N_shallow_max','xcor_N_shallow_mean','xcor_Z_deep_max','xcor_Z_deep_mean','xcor_Z_shallow_max','xcor_Z_shallow_mean','xcor_E_MA250_deep_max','xcor_E_MA250_deep_mean','xcor_E_MA250_shallow_max','xcor_E_MA250_shallow_mean','xcor_N_MA250_deep_max','xcor_N_MA250_deep_mean','xcor_N_MA250_shallow_max','xcor_N_MA250_shallow_mean','xcor_Z_MA250_deep_max','xcor_Z_MA250_deep_mean','xcor_Z_MA250_shallow_max','xcor_Z_MA250_shallow_mean','xcor_E_MA1000_deep_max','xcor_E_MA1000_deep_mean','xcor_E_MA1000_shallow_max','xcor_E_MA1000_shallow_mean','xcor_N_MA1000_deep_max','xcor_N_MA1000_deep_mean','xcor_N_MA1000_shallow_max','xcor_N_MA1000_shallow_mean','xcor_Z_MA1000_deep_max','xcor_Z_MA1000_deep_mean','xcor_Z_MA1000_shallow_max','xcor_Z_MA1000_shallow_mean','E_env_sim_deep_max','E_env_sim_deep_mean','E_env_sim_shallow_max','E_env_sim_shallow_mean','N_env_sim_deep_max','N_env_sim_deep_mean','N_env_sim_shallow_max','N_env_sim_shallow_mean','Z_env_sim_deep_max','Z_env_sim_deep_mean','Z_env_sim_shallow_max','Z_env_sim_shallow_mean']]