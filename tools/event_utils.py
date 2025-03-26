import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
import tqdm
import time
import sys
import math
from typing import Dict, Tuple

import h5py
import hdf5plugin   # must be imported
from numba import jit
import numpy as np

from tools.pose_utils import *

########################
# Event Loss / Luma Conversion
########################
def rgb_to_luma(rgb, esim=True):
    """
    Input:
    :rgb torch.Tensor (N_evs, 3)

    Output:
    :luma torch.Tensor (N_evs, 1)
    :esim Use luma-conversion-coefficients from esim, else from v2e-paper.
    "ITU-R recommendation BT.709 digital video (linear, non-gamma-corrected) color space conversion"
    see https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a
    or https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/ for numbers
    """

    device = rgb.device

    if esim:
        #  https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/imp/imp_opengl_renderer/src/opengl_renderer.cpp#L319-L321
        #  image format esim: https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/esim_visualization/src/ros_utils.cpp#L29-L36
        #  color conv factorsr rgb->gray: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
        r = 0.299
        g = 0.587
        b = 0.114
    else:
        r = 0.2126
        g = 0.7152
        b = 0.0722

    factors = torch.Tensor([r, g, b]).to(device)  # (3)
    luma = torch.sum(rgb * factors[None, :], axis=-1)  # (N_evs, 3) * (1, 3) => (N_evs)
    return luma[..., None]  # (N_evs, 1)

def lin_log(color, linlog_thres=20):
    """
    Input: 
    :color torch.Tensor of (N_rand_events, 1 or 3). 1 if use_luma, else 3 (rgb).
           We pass rgb here, if we want to treat r,g,b separately in the loss (each pixel must obey event constraint).
    """
    # Compute the required slope for linear region (below luma_thres)
    # we need natural log (v2e writes ln and "it comes from exponential relation")
    lin_slope = np.log(linlog_thres) / linlog_thres

    # Peform linear-map for smaller thres, and log-mapping for above thresh
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
    return lin_log_rgb


def estimate_C_thres_from_pol_dL(sum_pol_at_events, delta_linlog_intensity, esim=True):
    """
    This function is used for debugging/sanity checking only. 

    Input: 
    :sum_pol_at_events (N_rand_events, 1)
    :delta_linlog_intensity (N_rand_events, 1 or 3)

    Output: 
    :dict {"median_pos", "median_neg", "median_on_sign", "median_off_sign"}
    """
    if delta_linlog_intensity.shape[0] == 3:
        delta_linlog_intensity = rgb_to_luma(delta_linlog_intensity, esim=esim)

    mask_pos = torch.where((sum_pol_at_events[:, 0] > 0))[0]
    Cs_est_pos = delta_linlog_intensity[mask_pos] / sum_pol_at_events[mask_pos]
    mask_pos = torch.where((sum_pol_at_events[:, 0] > 0) & (delta_linlog_intensity[:, 0] >= 0))[0]
    Cs_est_pos_sign = delta_linlog_intensity[mask_pos] / sum_pol_at_events[mask_pos]

    mask_neg = torch.where((sum_pol_at_events[:, 0] < 0))[0]
    Cs_est_neg = delta_linlog_intensity[mask_neg] / sum_pol_at_events[mask_neg]
    mask_neg = torch.where((sum_pol_at_events[:, 0] < 0) & (delta_linlog_intensity[:, 0] <= 0))[0]
    Cs_est_neg_sign = delta_linlog_intensity[mask_neg] / sum_pol_at_events[mask_neg]

    if Cs_est_pos.shape[0] < 1:
        Cs_est_pos = torch.Tensor([0])
    if Cs_est_neg.shape[0] < 1:
        Cs_est_neg = torch.Tensor([0])
    if Cs_est_pos_sign.shape[0] < 1:
        Cs_est_pos_sign = torch.Tensor([0])
    if Cs_est_neg_sign.shape[0] < 1:
        Cs_est_neg_sign = torch.Tensor([0])

    return {
        "median_on": torch.median(Cs_est_pos),
        "median_off": torch.median(Cs_est_neg),
        "median_on_sign": torch.median(Cs_est_pos_sign),
        "median_off_sign": torch.median(Cs_est_neg_sign),
    }


########################
# Event Batches Helpers
########################
def check_evs_coord_range(event_batches, W=1280, H=720):
    """
    Input
    :event_batches list of event batches with (x, y, ts_ns, pol, [0, sum_pol])
    """

    for i, ev_batch in enumerate(event_batches):
        if not np.all((ev_batch[:, 0] < W) & (ev_batch[:, 1] < H) & (ev_batch[:, 0] >= 0) & (ev_batch[:, 1] >= 0)):
            print("Wrong image dimensions")
            sys.exit()


def should_transform_pol(event_batches, idx_pol=3):
    """
    Input
    :event_batches list of event batches with (x, y, ts_ns, pol, [0, sum_pol])
    """
    transform_pol = True
    for i, ev_batch in enumerate(event_batches):
        if np.any((ev_batch[:, idx_pol] == -1)):
            transform_pol = False
            break

    return transform_pol


def transform_pol(event_batches):
    """
    Input
    :event_batches list of event batches with (x, y, ts_ns, pol, [0, sum_pol])
    """
    mask = (1.0, 1.0, 1.0, 2.0, 1.0)
    event_batches = [ev * mask for ev in event_batches]
    mask = (0, 0, 0, -1, 0)
    event_batches = [ev + mask for ev in event_batches]
    print("loaded %s event_batches " % len(event_batches))
    return event_batches


def zero_pad_col_ev_batches(event_batches):
    """
    Input
    :event_batches list of event batches with (x, y, ts_ns, pol, [0, sum_pol])
    """
    for i, batch in enumerate(event_batches):
        num_evs_batch = batch.shape[0]
        event_batches[i] = np.hstack((batch, np.zeros((num_evs_batch, 1))))
    return event_batches


def check_evs_pol(event_batches, pol_neg=-1, pol_pos=1, idx_pol=3):
    """
    Input
    :event_batches list of event batches with (x, y, ts_ns, pol, [0, sum_pol])
    """
    for ev_batch in event_batches:
        assert np.all((ev_batch[:, idx_pol] == pol_pos) ^ (ev_batch[:, idx_pol] == pol_neg))
    return True


def check_evs_shapes(event_batches, tuple_size=5):
    """
    Input
    :event_batches list of event batches with (x, y, ts_ns, pol, [0, sum_pol])
    """
    for ev_batch in event_batches:
        assert ev_batch.shape[1] == tuple_size
    return True


##############
# Event Stats
##############
def get_evs_dictionary_mtNevs(evs_batches_ns, more_than=1):
    all_evs = np.concatenate(evs_batches_ns, axis=0)

    evs_dicts_xy_ns = {}
    for ev in all_evs:
        key_xy = (ev[0], ev[1])
        if key_xy in evs_dicts_xy_ns.keys():
            evs_dicts_xy_ns[key_xy].append(ev.tolist())
        else:
            evs_dicts_xy_ns[key_xy] = [ev.tolist()]

    # filter dictonary s.t. > 1 ev per pixel
    evs_dicts_xy_ns = dict((k, v) for k, v in evs_dicts_xy_ns.items() if len(v) > more_than) 
    evs_batches_ns.clear()
    del evs_batches_ns
    del all_evs
    all_evs = 0

    # precompute keys and num_elements for dictionary entries
    xys_mtNevs = list(evs_dicts_xy_ns.keys())
    num_evs_at_xy = np.asarray([len(evs_dicts_xy_ns[xy]) for xy in xys_mtNevs])
    xys_mtNevs = np.asarray(xys_mtNevs).astype(np.uint32)

    assert np.all(num_evs_at_xy > more_than)
    return evs_dicts_xy_ns, xys_mtNevs, num_evs_at_xy


@jit 
def read_window_h5(ef, key="x", start_idx=0, end_idx=10000000):
    N = (end_idx-start_idx)
    arr = np.zeros(N)
    # num_chunks = int(N/chunk_size)
    arr = ef[key][start_idx:end_idx]

    return arr

# from https://github.com/uzh-rpg/DSEC/blob/main/scripts/utils/eventslicer.py
class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        if 'events/x' in self.h5f.keys():
            for dset_str in ['p', 'x', 'y', 't']:
                self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]
        else:
            for dset_str in ['p', 'x', 'y', 't']:
                self.events[dset_str] = self.h5f['{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms = np.maximum(t_start_ms, 0)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        #if t_end_ms_idx is None:
        #    t_end_ms_idx = self.ms2idx(t_end_ms-1)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


def compute_ms_to_idx(tss_ns, ms_start=0):
    """
    evs_ns: (N, 4)
    idx_start: Integer
    ms_start: Integer
    """

    ms_to_ns = 1000000
    # tss_sorted, _ = torch.sort(tss_ns) 
    # assert torch.abs(tss_sorted != tss_ns).sum() < 500

    ms_end = int(math.floor(tss_ns.max()) / ms_to_ns)
    assert ms_end >= ms_start
    ms_window = np.arange(ms_start, ms_end + 1, 1).astype(np.uint64)
    ms_to_idx = np.searchsorted(tss_ns, ms_window * ms_to_ns, side="left", sorter=np.argsort(tss_ns))
    
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]] >= ms*ms_to_ns) for ms in ms_window]))
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]-1] < ms*ms_to_ns) for ms in ms_window if ms_to_idx[ms] >= 1]))
    
    return ms_to_idx