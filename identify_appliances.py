import ApplianceFeatures as ap
import numpy as np
from scipy import ndimage, signal
import sys
import pandas as pd
from datetime import timedelta

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def measure_jump(vals, loc, loc_end=0, search_width=1.0, jump_buffer=2):
    """
    Find the difference when going from one state to another.
    search_width is in seconds.
    """
    search_width_ticks = ceil(search_width/0.1664888858795166)
    previous_plateau = vals[loc - jump_buffer - search_width_ticks:loc - jump_buffer]
    after_plateau = vals[loc_end + jump_buffer:loc_end + jump_buffer + search_width_ticks]
    previous_mean = median(previous_plateau)
    previous_std = previous_plateau.std()
    after_mean = median(after_plateau)
    after_std = after_plateau.std()
    jump = after_mean - previous_mean
    jump_std = (after_std**2 + previous_std**2)**0.5
    return jump, jump_std


# Median Absolute Deviation clipping for input array of numbers.
def mad_clipping(input_data, sigma_clip_level):
    medval = np.median(input_data)
    sigma = 1.48 * np.median(abs(medval - input_data))
    high_sigma_clip_limit = medval + sigma_clip_level * sigma
    low_sigma_clip_limit = medval - sigma_clip_level * sigma
    clipped_data = input_data[(input_data>(low_sigma_clip_limit)) & (input_data<(high_sigma_clip_limit))]
    new_medval = np.median(clipped_data)
    new_sigma = 1.48 * np.median(abs(medval - clipped_data))
    return clipped_data, new_medval, new_sigma

def test_jumps_cleanliness(jumps):
    """
    Returns first flag True if all the jumps are clean, False otherwise
    Returns second flag True if jumps is empty, False otherwise
    """
    if len(jumps) == 0:
        return True, True
    for jump in jumps:
        if abs(jump[1][0]) < 5*jump[1][1]:
            return False, False
    else:
        return True, False

def combine_jumps_together_smarter(jumps, time_ticks, vals, thresh=30):
    # thresh is the num of indices between the end of one window 
    # and the beginning of the next to be considered the same
    # main df
    
    start_loc_list = []
    stop_loc_list = []
    values_list = []
    for jump in jumps:
        start_loc_list.append(jump[0][0])
        stop_loc_list.append(jump[0][1])
        values_list.append(jump[1][0])
    adf=pd.DataFrame({"start":start_loc_list,"stop":stop_loc_list,"values":values_list})
    # make blank output df
    new_df=pd.DataFrame(columns=["start","stop","values","count","sum"])
    # loop through and search
    columns=['start','stop','values']

    
    for index,row in adf.iterrows():
        found_neighbor = new_df.loc[(new_df.start < row['start']) & (new_df.stop + thresh > row['start'])]
        if len(found_neighbor) == 1:
            # print "found a neighbor"
            intind = int(found_neighbor.index)
            new_df.loc[intind]['stop'] = max(row['stop'],found_neighbor['stop'])
            new_df.loc[intind]['sum'] = row['values'] + new_df.loc[intind]['sum']
            new_df.loc[intind]['count'] += 1
            new_df.loc[intind]['values'] = new_df.loc[intind]['sum']/new_df.loc[intind]['count']
        elif len(found_neighbor) == 0:
            # print "adding new row"
            new_df = new_df.append(row.append(pd.Series({"sum":row['values'],"count":1})),ignore_index=True)
            row.append
        else:
            print "We have a problem. (In combined_jumps_together_smarter)"
    combined_jumps = []
    for entry in new_df.values:
        jump_interval = [entry[0], entry[1]]
        jump_height, jump_std = measure_jump(vals, jump_interval[0], loc_end=jump_interval[-1]-jump_interval[0], search_width=1.0, jump_buffer=6)
        if (abs(jump_height) > 5*jump_std) and (abs(jump_height)>20):
            combined_jumps.append([(int(jump_interval[0]), int(jump_interval[-1])), (jump_height, jump_std), (time_ticks[jump_interval[0]], time_ticks[jump_interval[-1]])])
    return combined_jumps
    

def plot_jumps(stream, datafilename, line_name, plot_flag=False):
    # L1 and L2 data is collected at a rate of 6.0064028254118895 times per second
    # HF spectra data is collected at a rate of 0.9375005859378663 times per second
    median_stream = ndimage.filters.median_filter(stream, 6.0064028254118895) # smooth with width 1 second of data
    smooth_stream = ndimage.filters.gaussian_filter1d(median_stream, 1.0) # smooth
    stream_gradient = np.gradient(smooth_stream)
#     peak_locs = signal.find_peaks_cwt(abs(stream_gradient), np.array([1]), min_snr=1)
#     peak_times = stream.index[peak_locs]
# signal.find_peaks_cwt was taking too long on the 24-hour data streams
# So, reverting to using simple median clipping to identify gradient vals above some threshold.
    clipped_stream_gradient, gradient_median, gradient_std = mad_clipping(stream_gradient, 3)
    peak_locs = np.where((abs(stream_gradient - gradient_median) > 10*gradient_std))[0]
    peak_times = stream.index[peak_locs]
    jumps = []
    sample_time_length = timedelta(seconds=3.5)
    buffer_time = timedelta(seconds=1.5)
    for loc in peak_locs:
        jump_difference, jump_std = measure_jump(smooth_stream, loc, loc_end=0, search_width=1.0, jump_buffer=6)
        if (abs(jump_difference) > 5* jump_std) and (abs(jump_difference) > 20):
            jumps.append([[loc, loc], (jump_difference, jump_std), (stream.index[loc], stream.index[loc])])
    if jumps[0][0][0] < 5:
        jumps.pop(0)
    if jumps[-1][0][0] > len(stream)-5:
        jumps.pop(-1)
    combined_jumps = combine_jumps_together_smarter(jumps, stream.index, smooth_stream, thresh=15)
    if plot_flag:
        stream_time_length = (stream.index[-1] - stream.index[0]).total_seconds()
        n_plot_rows = int(ceil(stream_time_length/1800.))
        fig, axes = plt.subplots(n_plot_rows, 1, figsize=(40,n_plot_rows*5))
        for n in range(n_plot_rows):
            stream_time_start = stream.index[0] + timedelta(seconds=1800*n)
            stream_time_end = stream.index[0] + timedelta(seconds=1800*(n+1))
            stream_trimmed_values = smooth_stream[(stream.index>stream_time_start)&(stream.index<stream_time_end)]
            stream_trimmed_timeticks = stream.index[(stream.index>stream_time_start)&(stream.index<stream_time_end)]
            axes[n].plot(stream_trimmed_timeticks, stream_trimmed_values, c="purple", lw=2)
            for jump in combined_jumps:
                if jump[2][1] > stream_time_start and jump[2][0] < stream_time_end:
                    axes[n].plot([jump[2][1], jump[2][1]], [0, 1000+jump[1][0]], color="red")
                    axes[n].text(jump[2][1], 1000+jump[1][0], str(int(round(jump[1][0]))) + " +/- " + str(round(jump[1][1],2)), rotation=45, va="bottom", ha="left")
            axes[n].set_ylim(10,4000)
            axes[n].set_xlim(stream_time_start, stream_time_end)
            axes[n].set_ylabel("Line Amplitude")
            axes[n].set_xlabel("Timestamp")
        canvas = FigureCanvas(fig)
        canvas.print_figure("full_plots/" + datafilename.split(".")[0] + "_" + line_name + "_jumps.png", dpi=72, bbox_inches='tight')
        close("all")
    return combined_jumps







# Read in Electric Time Stream

# file_list = os.listdir("data/hdf5storage/")
# for datafilename in file_list:
#     if datafilename[-3:] != ".h5":
#         continue

datafilename = "H1_07-12_testing.h5"

a = ap.ElectricTimeStream("data/hdf5storage/" + datafilename)
# Extract Components on the ap object
a.ExtractComponents()

L1_Amp = a.l1.amp.sum(axis=1)
# L2_Amp = a.l2.amp.sum(axis=1)

L1_combined_jumps = plot_jumps(L1_Amp, datafilename, "L1", plot_flag=False)
# L2_combined_jumps = plot_jumps(L2_Amp, datafilename, "L2", plot_flag=False)


def measure_jump_features(time_stream, start_index, end_index, search_width=1.0, jump_buffer=6):
    features = []
    harmonics = ["0", "1", "2", "3", "4", "5"]
    for h in harmonics:
        features.append(measure_jump(time_stream.amp[h].values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
    return features

L1_features = measure_jump_features(a.l1, 388079, 388086)