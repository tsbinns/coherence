import numpy as np
import pandas as pd

np.random.seed(seed=0)

# Channels whose data will be plotted
channels = ["ECOG_L_1_SMC_AT", "ECOG_L_2_SMC_AT", "ECOG_L_3_SMC_AT",
            "ECOG_L_4_SMC_AT", "ECOG_L_5_SMC_AT", "ECOG_L_6_SMC_AT"]
# Frequency bands of the power data (each band will be on a separate glass brain)
freq_bands = np.tile(['alpha', 'beta', 'gamma'], (len(channels),1))
# The power of each frequency band in each channel
band_power = np.random.rand(len(channels), np.shape(freq_bands)[1])*10

# Packaged data
data = list(zip(channels, freq_bands, band_power))
data = pd.DataFrame(data, columns=['channel', 'freq_bands', 'band_power'])