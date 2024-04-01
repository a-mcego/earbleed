import numpy as np
from scipy.interpolate import interp1d

# Given tables from ISO 226:2003
# Augmented with extra values
table_frequencies = np.array([0, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 60000])
af = np.array([0.532, 0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315,
               0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243,
               0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301, 0.349])
Lu = np.array([-31.6, -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1,
               -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7,
               2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1, -6.2])
Tf = np.array([78.5, 78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4,
               11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2,
               -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3, 22.1])

# Function to calculate equal-loudness contour for a given loudness level (phon)
def equal_loudness_contour(phon_level,frequencies):
    if phon_level < 0 or phon_level > 90:
        raise ValueError("Phon level must be between 0 and 90.")
    
    # Linear interpolation functions
    af_interp = interp1d(table_frequencies, af, kind='linear')
    Lu_interp = interp1d(table_frequencies, Lu, kind='linear')
    Tf_interp = interp1d(table_frequencies, Tf, kind='linear')
    
    # Calculate sound pressure level (SPL) for each frequency
    SPL = phon_level + Lu_interp(frequencies) - af_interp(frequencies) * np.log10(0.4 * (10**(Tf_interp(frequencies) / 10 - 9)))
    
    return SPL

def get_freq_values(samplerate, windowsize):
    frequencies = np.linspace(0,samplerate*0.5,windowsize,endpoint=False)
    phon_level = 60
    spl = equal_loudness_contour(phon_level,frequencies)
    spl -= np.max(spl)
    return np.power(10.0, spl*0.05)
