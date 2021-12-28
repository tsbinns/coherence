from logging import raiseExceptions
import mne_bids
import numpy as np
from mne import read_annotations




class Signal:

    def __init__(self,
        verbose: bool = True):

        self._verbose = verbose
        self._processing_steps = {}


    def get_coordinates(self) -> list:

        return np.multiply(self.raw._get_channel_positions().copy().tolist(), 1000).tolist()


    def set_coordinates(self,
        coordinates: list,
        names: list):

        self.raw._set_channel_positions(coordinates, names)

        if self._verbose:
            print(f"Setting channel coordinates to:\n{coordinates}.")


    def get_data(self) -> np.array:

        return self.raw.get_data(reject_by_annotation='omit').copy()


    def load_raw(self,
        path_raw: mne_bids.BIDSPath):

        self._path_raw = path_raw
        self.raw = mne_bids.read_raw_bids(bids_path=self.path_raw, verbose=False)
        self.raw.load_data()

        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.")

    
    def load_annotations(self,
        path_annotations: str):

        self.raw.set_annotations(read_annotations(path_annotations))

        if self._verbose:
            print(f"Applying annotations to the data from the filepath:\n{path_annotations}.")

        self._processing_steps['annotated'] = True

    
    def pick_channels(self,
        channels: list):

        self.raw.pick_channels(channels)

        if self._verbose:
            print(f"Picking specified channels from the data.\nChannels: {channels}.")

        self._processing_steps['channel_picks'] = channels

    
    def bandpass_filter(self,
        frequencies: list):

        if len(frequencies) != 2:
            raise Exception(f"Error when bandpass filtering the data.\nTwo frequencies are required, but {len(frequencies)} are given.")

        self.raw.filter(frequencies[0], frequencies[1])

        if self._verbose:
            print(f"Bandpass filtering the data.\nLow frequency: {frequencies[0]} Hz. High frequency: {frequencies[1]} Hz.")

        self._processing_steps['bandpass_filter'] = frequencies

    
    def notch_filter(self,
        line_frequency: int):

        frequencies = np.arange(0, self.raw.info['lowpass']+1, line_frequency)
        self.raw.notch_filter(frequencies)

        if self._verbose:
            print(f"Notch filtering the data with line noise frequency {line_frequency} Hz at the following frequencies (Hz): {frequencies}.")

        self._processing_steps['notch_filter'] = frequencies


    def resample(self,
        resample_frequency: int):

        self.raw.resample(resample_frequency)

        if self._verbose:
            print(f"Resampling the data at {resample_frequency} Hz.")

        self._processing_steps['resample'] = resample_frequency


    def rereferencing(self):
        pass




