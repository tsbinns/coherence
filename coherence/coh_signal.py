import mne_bids
import numpy as np
from mne import read_annotations
from coh_rereference import RerefBipolar, RerefCAR, RerefNone




class Signal:

    def __init__(self,
        verbose: bool = True
        ):

        self._verbose = verbose
        self.processing_steps = []
        self.extra_info = {}


    def get_coordinates(self
        ) -> list:

        return self.raw._get_channel_positions().copy().tolist()


    def set_coordinates(self,
        ch_coords: list,
        ch_names: list
        ):

        self.raw._set_channel_positions(ch_coords, ch_names)

        if self._verbose:
            print(f"Setting channel coordinates to:\n{ch_coords}.")


    def get_data(self
        ) -> np.array:

        return self.raw.get_data(reject_by_annotation='omit').copy()


    def load_raw(self,
        path_raw: mne_bids.BIDSPath
        ):

        self._path_raw = path_raw
        self.raw = mne_bids.read_raw_bids(bids_path=self.path_raw, verbose=False)
        self.raw.load_data()

        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.")

    
    def load_annotations(self,
        path_annots: str
        ):

        self.raw.set_annotations(read_annotations(path_annots))

        if self._verbose:
            print(f"Applying annotations to the data from the filepath:\n{path_annots}.")

        self._annotations_added = True
        self.processing_steps.append(['annotations_added', True])


    def _pick_extra_info(self,
        ch_names: list
        ):

        for key in self.extra_info.keys():
            new_entry = {ch_name: self.extra_info[key][ch_name] for ch_name in ch_names}
            self.extra_info[key] = new_entry

    
    def pick_channels(self,
        ch_names: list
        ):

        self.raw.pick_channels(ch_names)
        self._pick_extra_info(ch_names)

        if self._verbose:
            print(f"Picking specified channels from the data.\nChannels: {ch_names}.")

        self.processing_steps.append(['channel_picks', ch_names])

    
    def bandpass_filter(self,
        freqs: list
        ):

        if len(freqs) != 2:
            raise Exception(f"Error when bandpass filtering the data.\nTwo frequencies are required, but {len(freqs)} are given.")

        self.raw.filter(freqs[0], freqs[1])

        if self._verbose:
            print(f"Bandpass filtering the data.\nLow frequency: {freqs[0]} Hz. High frequency: {freqs[1]} Hz.")

        self._bandpass_filtered = True
        self.processing_steps['bandpass_filter'] = freqs

    
    def notch_filter(self,
        line_freq: int
        ):

        freqs = np.arange(0, self.raw.info['lowpass']+1, line_freq)
        self.raw.notch_filter(freqs)

        if self._verbose:
            print(f"Notch filtering the data with line noise frequency {line_freq} Hz at the following frequencies (Hz): {freqs}.")

        self._notch_filtered = True
        self.processing_steps['notch_filter'] = freqs


    def resample(self,
        resample_freq: int
        ):

        self.raw.resample(resample_freq)

        if self._verbose:
            print(f"Resampling the data at {resample_freq} Hz.")

        self._resampled = True
        self.processing_steps['resample'] = resample_freq


    def rereferencing(self,
        reref_settings: dict
        ):
        
        self._rereferenced = True




