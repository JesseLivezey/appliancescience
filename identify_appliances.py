import ApplianceFeatures as ap
import numpy as np
from scipy import ndimage, signal
import sys
import pandas as pd
import os

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import timedelta, datetime

def measure_jump(vals, loc, loc_end, search_width=1.0, jump_buffer=2):
    """
    Find the difference when going from one state to another.
    search_width is in seconds.
    """
    try:
        search_width_ticks = np.ceil(search_width/0.1664888858795166)
        previous_plateau = vals[loc - jump_buffer - search_width_ticks:loc - jump_buffer]
        after_plateau = vals[loc_end + jump_buffer:loc_end + jump_buffer + search_width_ticks]
        previous_mean = median(previous_plateau)
        previous_std = previous_plateau.std()
        after_mean = median(after_plateau)
        after_std = after_plateau.std()
        jump = after_mean - previous_mean
        jump_std = (after_std**2 + previous_std**2)**0.5
        if isnan(jump):
            return 0, 0
        else:
            return jump, jump_std
    except:
        return 0, 0


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
        jump_height, jump_std = measure_jump(vals, jump_interval[0], jump_interval[-1], search_width=2.5, jump_buffer=6)
        if (abs(jump_height) > 5*jump_std) and (abs(jump_height)>20):
            combined_jumps.append([(int(jump_interval[0]), int(jump_interval[-1])), (jump_height, jump_std), (time_ticks[jump_interval[0]], time_ticks[jump_interval[-1]])])
    return combined_jumps
    



def find_jumps(stream):
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
    min_peak_value = max(10*gradient_std, 1.5)
    peak_locs = np.where((abs(stream_gradient - gradient_median) > min_peak_value))[0]
    
    
    
    thresh=12
    start_loc_list = peak_locs
    stop_loc_list = peak_locs
    adf=pd.DataFrame({"start":start_loc_list,"stop":stop_loc_list})
    # make blank output df
    new_df=pd.DataFrame(columns=["start","stop","count"])
    # loop through and search
    columns=['start','stop']
    for index,row in adf.iterrows():
        found_neighbor = new_df.loc[(new_df.start < row['start']) & (new_df.stop + thresh > row['start'])]
        if len(found_neighbor) == 1:
            # print "found a neighbor"
            intind = int(found_neighbor.index)
            new_df.loc[intind]['stop'] = max(row['stop'],found_neighbor['stop'])
            new_df.loc[intind]['count'] += 1
        elif len(found_neighbor) == 0:
            # print "adding new row"
            new_df = new_df.append(row.append(pd.Series({"count":1})),ignore_index=True)
            row.append
        else:
            print "We have a problem. (In combined_jumps_together_smarter)"
    
    peak_locs = new_df.values[:,:2]
    jumps = []
    sample_time_length = timedelta(seconds=2.5)
    buffer_time = timedelta(seconds=1.0)
    for loc in peak_locs:
        jump_difference, jump_std = measure_jump(smooth_stream, loc[0], loc[1], search_width=2.5, jump_buffer=6)
        if (abs(jump_difference) > 5* jump_std) and (abs(jump_difference) > 10):
            jumps.append([[loc[0], loc[1]], (jump_difference, jump_std), (stream.index[loc[0]], stream.index[loc[1]])])
    if jumps[0][0][0] < 5:
        jumps.pop(0)
    if jumps[-1][0][0] > len(stream)-5:
        jumps.pop(-1)
    combined_jumps = combine_jumps_together_smarter(jumps, stream.index, smooth_stream, thresh=12)
    return combined_jumps


def plot_jumps(stream, datafilename, line_name):
    # L1 and L2 data is collected at a rate of 6.0064028254118895 times per second
    # HF spectra data is collected at a rate of 0.9375005859378663 times per second
    median_stream = ndimage.filters.median_filter(stream, 6.0064028254118895) # smooth with width 1 second of data
    smooth_stream = ndimage.filters.gaussian_filter1d(median_stream, 1.0) # smooth
    combined_jumps = find_jumps(stream)
    stream_time_length = (stream.index[-1] - stream.index[0]).total_seconds()
    n_plot_rows = int(np.ceil(stream_time_length/1800.))
    fig, axes = plt.subplots(n_plot_rows, 1, figsize=(40,n_plot_rows*5))
    for n in range(n_plot_rows):
        stream_time_start = stream.index[0] + timedelta(seconds=1800*n)
        stream_time_end = stream.index[0] + timedelta(seconds=1800*(n+1))
        stream_trimmed_values = smooth_stream[(stream.index>stream_time_start) & (stream.index<stream_time_end)]
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



def find_appliance_jumps(stream):
# L1 and L2 data is collected at a rate of 6.0064028254118895 times per second
# HF spectra data is collected at a rate of 0.9375005859378663 times per second
    try:
        median_stream = ndimage.filters.median_filter(stream, 6.0064028254118895) # smooth with width 1 second of data
        smooth_stream = ndimage.filters.gaussian_filter1d(median_stream, 1.0) # smooth
        stream_gradient = np.gradient(smooth_stream)
        peak_locs = signal.find_peaks_cwt(abs(stream_gradient), np.array([1]), min_snr=1)
        clipped_stream_gradient, stream_gradient_median, std_around_zero = mad_clipping(stream_gradient, 3)
        min_peak_thresh = max(abs(stream_gradient))/10.0
        vetted_peak_locs = list(np.where((abs(stream_gradient[peak_locs]) > 5*std_around_zero) & (abs(stream_gradient[peak_locs])>min_peak_thresh))[0])
        peak_indices = list(np.array(peak_locs)[vetted_peak_locs])
        peak_times = stream.index[peak_indices]
        peak_data = stream_gradient[peak_indices]
        # return peak_indices[np.where(peak_data==peak_data.max())[0][0]], peak_indices[np.where(peak_data==peak_data.min())[0][0]]
        
        thresh=60
        start_loc_list = peak_indices
        stop_loc_list = peak_indices
        adf=pd.DataFrame({"start":start_loc_list,"stop":stop_loc_list})
        # make blank output df
        new_df=pd.DataFrame(columns=["start","stop","count"])
        # loop through and search
        columns=['start','stop']
        for index,row in adf.iterrows():
            found_neighbor = new_df.loc[(new_df.start < row['start']) & (new_df.stop + thresh > row['start'])]
            if len(found_neighbor) == 1:
                # print "found a neighbor"
                intind = int(found_neighbor.index)
                new_df.loc[intind]['stop'] = max(row['stop'],found_neighbor['stop'])
                new_df.loc[intind]['count'] += 1
            elif len(found_neighbor) == 0:
                # print "adding new row"
                new_df = new_df.append(row.append(pd.Series({"count":1})),ignore_index=True)
                row.append
            else:
                print "We have a problem. (In combined_jumps_together_smarter)"
        on_jump_loc = new_df.values[0][0]
        on_jump_loc_end = new_df.values[0][1]
        
        off_jump_loc = new_df.values[-1][0]
        off_jump_loc_end = new_df.values[-1][1]
        
        return on_jump_loc, on_jump_loc_end, off_jump_loc, off_jump_loc_end
    except:
        return None, None, None, None

def measure_jump_features(time_stream, start_index, end_index, search_width=1.0, jump_buffer=6):
    features = []
#     harmonics = ["0", "1", "2", "3", "4", "5"]
#     for h in harmonics:
#         features.append(measure_jump(time_stream.real[h].values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
#         features.append(measure_jump(time_stream.imag[h].values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
#         features.append(measure_jump(time_stream.amp[h].values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
#         features.append(measure_jump(time_stream.pf[h].values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
    features.append(measure_jump(time_stream.real.sum(axis=1).values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
    features.append(measure_jump(time_stream.imag.sum(axis=1).values, start_index, end_index, search_width=search_width, jump_buffer=jump_buffer))
    return features







def extract_training_features():
    # Extract training features
    extracted_features_list = []
    for appliance_num in range(394):
        app=ap.Appliance(appliance_num)
        appliance_on_jump_L1_loc, appliance_on_jump_L1_loc_end, appliance_off_jump_L1_loc, appliance_off_jump_L1_loc_end = find_appliance_jumps(app.l1_event.amp.sum(axis=1))
        appliance_on_jump_L2_loc, appliance_on_jump_L2_loc_end, appliance_off_jump_L2_loc, appliance_off_jump_L2_loc_end = find_appliance_jumps(app.l2_event.amp.sum(axis=1))
        if appliance_on_jump_L1_loc and not appliance_on_jump_L2_loc:
            appliance_on_jump_L2_loc = appliance_on_jump_L1_loc
            appliance_off_jump_L2_loc = appliance_off_jump_L1_loc
            appliance_on_jump_L2_loc_end = appliance_on_jump_L1_loc_end
            appliance_off_jump_L2_loc_end = appliance_off_jump_L1_loc_end
        elif appliance_on_jump_L2_loc and not appliance_on_jump_L1_loc:
            appliance_on_jump_L1_loc = appliance_on_jump_L2_loc
            appliance_off_jump_L1_loc = appliance_off_jump_L2_loc
            appliance_on_jump_L1_loc_end = appliance_on_jump_L2_loc_end
            appliance_off_jump_L1_loc_end = appliance_off_jump_L2_loc_end
        appliance_on_features_L1 = measure_jump_features(app.l1_event, appliance_on_jump_L1_loc, appliance_on_jump_L1_loc_end)
        appliance_off_features_L1 = measure_jump_features(app.l1_event, appliance_off_jump_L1_loc, appliance_off_jump_L1_loc_end)
        appliance_on_features_L2 = measure_jump_features(app.l2_event, appliance_on_jump_L2_loc, appliance_on_jump_L2_loc_end)
        appliance_off_features_L2 = measure_jump_features(app.l2_event, appliance_off_jump_L2_loc, appliance_off_jump_L2_loc_end)
        # require that the on and off power changes are similar in absolute value
#         L1_real_discrepancy = abs(abs(appliance_on_features_L1[0][0]) - abs(appliance_off_features_L1[0][0]))/abs(abs(appliance_on_features_L1[0][0]) + abs(appliance_off_features_L1[0][0])/2.0)
#         L1_imag_discrepancy = abs(abs(appliance_on_features_L1[1][0]) - abs(appliance_off_features_L1[1][0]))/abs(abs(appliance_on_features_L1[1][0]) + abs(appliance_off_features_L1[1][0])/2.0)
#         L2_real_discrepancy = abs(abs(appliance_on_features_L2[0][0]) - abs(appliance_off_features_L2[0][0]))/abs(abs(appliance_on_features_L2[0][0]) + abs(appliance_off_features_L2[0][0])/2.0)
#         L2_imag_discrepancy = abs(abs(appliance_on_features_L2[1][0]) - abs(appliance_off_features_L2[1][0]))/abs(abs(appliance_on_features_L2[1][0]) + abs(appliance_off_features_L2[1][0])/2.0)
#         discrepancy_limit=0.1
#         if L1_real_discrepancy<discrepancy_limit and L1_imag_discrepancy<discrepancy_limit and L2_real_discrepancy<discrepancy_limit and L2_imag_discrepancy<discrepancy_limit:
#             extracted_features_list.append([app.house, app.id, app.name, "on", appliance_on_features_L1[0][0], appliance_on_features_L1[1][0], appliance_on_features_L2[0][0], appliance_on_features_L2[1][0]])
#             extracted_features_list.append([app.house, app.id, app.name, "off", appliance_off_features_L1[0][0], appliance_off_features_L1[1][0], appliance_off_features_L2[0][0], appliance_off_features_L2[1][0]])
        if max(appliance_on_features_L1[0][0], appliance_on_features_L1[1][0], appliance_on_features_L2[0][0], appliance_on_features_L2[1][0])<30 or min(appliance_off_features_L1[0][0], appliance_off_features_L1[1][0], appliance_off_features_L2[0][0], appliance_off_features_L2[1][0])>-30:
            continue
        else:
            extracted_features_list.append([app.house, app.id, app.name, "on", appliance_on_features_L1[0][0], appliance_on_features_L1[1][0], appliance_on_features_L2[0][0], appliance_on_features_L2[1][0]])
            extracted_features_list.append([app.house, app.id, app.name, "off", appliance_off_features_L1[0][0], appliance_off_features_L1[1][0], appliance_off_features_L2[0][0], appliance_off_features_L2[1][0]])
        
        
    extracted_features_array = np.array(extracted_features_list)
    training_id_list = list(set(extracted_features_array[:,1].astype(float)))
    training_house_list = ["H1", "H2", "H3", "H4"]
    
    merged_features_list = []
    # Combine multiple instances of a training example into one generic feature.
    # Intelligently exclude outliers
    for house in training_house_list:
        for id_num in training_id_list:
            training_features_subset = []
            for trainf in extracted_features_list:
                # First do it for "on" transitions
                if (trainf[0]==house) and (trainf[1]==id_num) and (trainf[3]=="on"):
                    training_features_subset.append(trainf)
            if len(training_features_subset) > 2:
                training_features_subset_array = np.array(training_features_subset)
                new_house = training_features_subset_array[0][0]
                new_id = training_features_subset_array[0][1]
                new_name = training_features_subset_array[0][2]
                new_transition = training_features_subset_array[0][3]
                new_L1_real = median(training_features_subset_array[:,4].astype(float))
                new_L1_imag = median(training_features_subset_array[:,5].astype(float))
                new_L2_real = median(training_features_subset_array[:,6].astype(float))
                new_L2_imag = median(training_features_subset_array[:,7].astype(float))
                new_feature = [ new_house,
                                new_id,
                                new_name,
                                new_transition,
                                new_L1_real,
                                new_L1_imag,
                                new_L2_real,
                                new_L2_imag]
                merged_features_list.append(new_feature)
            elif len(training_features_subset) == 2:
                merged_features_list.append(training_features_subset[0])
                merged_features_list.append(training_features_subset[1])
            elif len(training_features_subset) == 1:
                merged_features_list.append(training_features_subset[0])
            
            training_features_subset = []
            for trainf in extracted_features_list:
                # First do it for "on" transitions
                if (trainf[0]==house) and (trainf[1]==id_num) and (trainf[3]=="off"):
                    training_features_subset.append(trainf)
            if len(training_features_subset) > 2:
                training_features_subset_array = np.array(training_features_subset)
                new_house = training_features_subset_array[0][0]
                new_id = training_features_subset_array[0][1]
                new_name = training_features_subset_array[0][2]
                new_transition = training_features_subset_array[0][3]
                new_L1_real = median(training_features_subset_array[:,4].astype(float))
                new_L1_imag = median(training_features_subset_array[:,5].astype(float))
                new_L2_real = median(training_features_subset_array[:,6].astype(float))
                new_L2_imag = median(training_features_subset_array[:,7].astype(float))
                new_feature = [ new_house,
                                new_id,
                                new_name,
                                new_transition,
                                new_L1_real,
                                new_L1_imag,
                                new_L2_real,
                                new_L2_imag]
                merged_features_list.append(new_feature)
            elif len(training_features_subset) == 2:
                merged_features_list.append(training_features_subset[0])
                merged_features_list.append(training_features_subset[1])
            elif len(training_features_subset) == 1:
                merged_features_list.append(training_features_subset[0])
    
    training_features_file = file("simple_training_features.csv", "w")
    training_features_file.write("house,id,name,transition,L1_real,L1_imag,L2_real,L2_imag\n")
    for mf in merged_features_list:
        training_features_file.write(mf[0] + "," + str(mf[1]) + "," + mf[2] + "," + mf[3] + "," + 
            str(mf[4]) + "," + 
            str(mf[5]) + "," + 
            str(mf[6]) + "," + 
            str(mf[7]) + "\n") 
    training_features_file.close()





# Read in Electric Time Stream

file_list = os.listdir("data/hdf5storage/")

# file_list = ["H1_04-13.h5"]

# for datafilename in file_list:
#     if datafilename[-3:] != ".h5":
#         continue

datafilename = sys.argv[1]

# datafilename = "H1_07-11_testing.h5"

a = ap.ElectricTimeStream("data/hdf5storage/" + datafilename)
# Extract Components on the ap object
a.ExtractComponents()

L1_Amp = a.l1.amp.sum(axis=1)
L2_Amp = a.l2.amp.sum(axis=1)

L1_combined_jumps = find_jumps(L1_Amp)
L2_combined_jumps = find_jumps(L2_Amp)

combined_jumps_indices = []
for jump in L1_combined_jumps:
    combined_jumps_indices.append(jump[0])
for jump in L2_combined_jumps:
    combined_jumps_indices.append(jump[0])
combined_jumps_indices.sort()
testing_features_file = file("simple_features_" + datafilename[:-3] + ".csv", "w")
testing_features_file.write("timestamp,L1_real,L1_imag,L2_real,L2_imag\n")
for jump in combined_jumps_indices:
    timestamp = a.l1.index[int((jump[1]+jump[0])/2.0)]
    L1_features = measure_jump_features(a.l1, jump[0], jump[1])
    L2_features = measure_jump_features(a.l2, jump[0], jump[1])
    if (L1_features[0][0]**2 + L1_features[1][0]**2)**0.5 > (L2_features[0][0]**2 + L2_features[1][0]**2)**0.5:
        if L1_features[0][0] + L1_features[1][0] > 0:
            timestamp = a.l1.index[jump[0]]
        else:
            timestamp = a.l1.index[jump[1]]
    else:
        if L2_features[0][0] + L2_features[1][0] > 0:
            timestamp = a.l1.index[jump[0]]
        else:
            timestamp = a.l1.index[jump[1]]
    seconds = (timestamp - datetime(1970,1,1)).total_seconds()
    testing_features_file.write(str(seconds) + "," + 
        str(L1_features[0][0]) + "," + 
        str(L1_features[1][0]) + "," + 
        str(L2_features[0][0]) + "," + 
        str(L2_features[1][0]) + "\n")
testing_features_file.close()




