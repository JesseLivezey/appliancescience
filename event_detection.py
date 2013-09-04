import numpy as np
from matplotlib.pylab import *
from scipy import io
from cmath import phase
import sys

import pandas as pd

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy import ndimage, signal
from scipy.cluster.vq import kmeans2

# This function parses the weirdly-formatted tagged info into a more friendly
# dictionary. "tagged info" are time ranges for when an identified appliance is
# turned off (it is one in between the times given in the OnOffSeq)
def parse_tagging_info(tagging_info_buffer):
    tagging_info_dict = {}
    for n in range(len(tagging_info_buffer)):
        id = tagging_info_buffer[n][0][0][0].astype(int)
        name = tagging_info_buffer[n][1][0][0][0].astype(str)
        on_time = tagging_info_buffer[n][2][0][0].astype(int)
        off_time = tagging_info_buffer[n][3][0][0].astype(int)
        if id not in tagging_info_dict.keys():
            tagging_info_dict[id] = {}
            tagging_info_dict[id]["ApplianceName"] = name
            tagging_info_dict[id]["OnOffSeq"] = [(on_time, off_time)]
        else:
            tagging_info_dict[id]["OnOffSeq"].append( (on_time, off_time) )
    return tagging_info_dict

def measure_jump(vals, loc, loc_end=0, search_width=1.0, jump_buffer=2):
    """
    Find the difference when going from one state to another.
    search_width is in seconds.
    """
    search_width_ticks = ceil(search_width/0.1664888858795166)
    previous_plateau = vals[loc - jump_buffer - search_width_ticks:loc - jump_buffer]
    after_plateau = vals[loc_end + jump_buffer:loc_end + jump_buffer + search_width_ticks]
    previous_mean = previous_plateau.mean()
    previous_std = previous_plateau.std()
    after_mean = after_plateau.mean()
    after_std = after_plateau.std()
    jump = after_mean - previous_mean
    jump_std = (after_std**2 + previous_std**2)**0.5
    return jump, jump_std

def merge_jumps_together(jump_interval_indices):
    merged_jumps = []
    for n in range(len(jump_interval_indices)):
        if n > 0 and (jump_interval_indices[n][0] - 6) < jump_interval_indices[n-1][-1]:
                merged_jump_interval = [jump_interval_indices[n-1][-1], jump_interval_indices[n][-1]]
                if merged_jump_interval not in merged_jumps:
                    merged_jumps.append(merged_jump_interval)
        if n < (len(jump_interval_indices)-1) and jump_interval_indices[n][-1] > (jump_interval_indices[n+1][0] - 6):
                merged_jump_interval = [jump_interval_indices[n][0], jump_interval_indices[n+1][-1]]
                if merged_jump_interval not in merged_jumps:
                    merged_jumps.append(merged_jump_interval)
        else:
            merged_jump_interval = [jump_interval_indices[n][0], jump_interval_indices[n][-1]]
            if merged_jump_interval not in merged_jumps:
                merged_jumps.append(merged_jump_interval)
    indices_to_nix = []
    indices_stored = []
    for n in range(len(merged_jumps)):
        for m in range(len(merged_jumps)):
            if merged_jumps[n][0] <= merged_jumps[m][0] and merged_jumps[n][-1] >= merged_jumps[m][-1]:
                if m not in indices_stored:
                    indices_stored.append(m)
                else:
                    indices_to_nix.append(m)
    trimmed_merged_jumps = []
    for p in range(len(merged_jumps)):
        if p not in indices_to_nix:
            trimmed_merged_jumps.append(merged_jumps[p])
    return trimmed_merged_jumps





def combine_jumps_together_smarter(jumps, time_ticks, smooth_vals):
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

    thresh = 30 # between the end of one window and the beginning of the next to be considered the same
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
        jump_height, jump_std = measure_jump(smooth_vals, jump_interval[0], loc_end=jump_interval[-1]-jump_interval[0], search_width=1.5, jump_buffer=5)
        if abs(jump_height) > 3*jump_std:
            combined_jumps.append([(jump_interval[0], jump_interval[-1]), (jump_height, jump_std), (time_ticks[jump_interval[0]], time_ticks[jump_interval[-1]])])
    return combined_jumps
    
    
    

def combine_jumps_together(jumps, time_ticks, smooth_vals):    
    # Combine jumps with large std into jumps with start and stop times
    jump_interval_indices = []
    current_jump_interval = set()
    n = 0
    while n < len(jumps):
        if abs(jumps[n][1][0]) > 5*jumps[n][1][1]: # if it's a clean jump
            current_jump_interval = current_jump_interval.union({jumps[n][0][0]})
            current_jump_interval = current_jump_interval.union({jumps[n][0][1]})
            jump_interval_indices.append(current_jump_interval)
            current_jump_interval = set()
            # print "a, clean jump", jumps[n][0][0], jumps[n][0][1]
        elif (n<len(jumps)-1) and (jumps[n+1][0][0] - jumps[n][0][1]) > 30:
            current_jump_interval = current_jump_interval.union({jumps[n][0][0]})
            current_jump_interval = current_jump_interval.union({jumps[n][0][1]})
            jump_interval_indices.append(current_jump_interval)
            current_jump_interval = set()
        elif (n>0) and (jumps[n][0][0] - jumps[n-1][0][1]) > 30:
            current_jump_interval = current_jump_interval.union({jumps[n][0][0]})
            current_jump_interval = current_jump_interval.union({jumps[n][0][1]})
            jump_interval_indices.append(current_jump_interval)
            current_jump_interval = set()
        elif (n>0) and (n<len(jumps)-1) and (jumps[n][0][0] - jumps[n-1][0][1] < jumps[n+1][0][0] - jumps[n][0][1]) and len(jump_interval_indices)>0:
            jump_interval_indices[-1] = jump_interval_indices[-1].union(current_jump_interval, {jumps[n][0][0], jumps[n][0][1]})
            current_jump_interval = set()
            # print "b, combine down", jumps[n][0][0], jumps[n][0][1]
        else:
            current_jump_interval = current_jump_interval.union({jumps[n][0][0]})
            current_jump_interval = current_jump_interval.union({jumps[n][0][1]})
            # print "c, combine up", jumps[n][0][0], jumps[n][0][1]
        n += 1
        # print jump_interval_indices
    
    if len(current_jump_interval) == len(jumps):
        return []

    if len(current_jump_interval) > 0 and len(jump_interval_indices)>0:
        jump_interval_indices[-1] = jump_interval_indices[-1].union(current_jump_interval)
    
    elif len(current_jump_interval) > 0:
        jump_interval_indices = [current_jump_interval]

    listed_jump_interval_indices = []
    for element in jump_interval_indices:
        sorted_element = sorted(list(element))
        listed_jump_interval_indices.append(sorted_element)
    
    merged_jump_indices = merge_jumps_together(listed_jump_interval_indices)
    
    combined_jumps = []
    for jump_interval in merged_jump_indices:
        jump_height, jump_std = measure_jump(smooth_vals, jump_interval[0], loc_end=jump_interval[-1]-jump_interval[0], search_width=1.5, jump_buffer=5)
        if abs(jump_height) > 3*jump_std:
            combined_jumps.append([(jump_interval[0], jump_interval[-1]), (jump_height, jump_std), (time_ticks[jump_interval[0]], time_ticks[jump_interval[-1]])])
    
    return combined_jumps



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
    for n in range(len(jumps)):
        if (n<len(jumps)-1) and (jumps[n+1][0][0] - jumps[n][0][1]) > 30:
            return False, False
        if (n>0) and (jumps[n][0][0] - jumps[n-1][0][1]) > 30:
            return False, False
    else:
        return True, False



def event_detector2(time_ticks, vals):
    confidence_flag = False
    # smooth with a width of 1 second
    smooth_vals = ndimage.filters.gaussian_filter1d(vals, 0.5/(time_ticks[1]-time_ticks[0]))

    # take the gradient to find where the changes occur
    smooth_detection_stream = np.gradient(smooth_vals)
    std_around_zero = smooth_detection_stream[abs(smooth_detection_stream)<0.5].std()
    smooth_detection_stream = where(abs(smooth_detection_stream)<5*std_around_zero, 0, smooth_detection_stream)
    peak_locs = signal.find_peaks_cwt(abs(smooth_detection_stream), np.array([1]), min_snr=0.05)
    
    # vetted_peak_locs = list(where(abs(smooth_detection_stream[peak_locs]) > 1.5)[0])
    vetted_peak_locs = list(where(abs(smooth_detection_stream[peak_locs]) > 5*std_around_zero)[0])
    
    peak_indices = list(array(peak_locs)[vetted_peak_locs])
    peak_times = time_ticks[peak_indices]
    peak_data = smooth_detection_stream[peak_indices]

    jumps = []
    for loc in peak_indices:
        if loc > 8 and loc < smooth_vals.size-8:
            jump_height, jump_std = measure_jump(smooth_vals, loc, loc_end=0, search_width=1.0, jump_buffer=5)
            jumps.append([[loc, loc], (jump_height, jump_std), (time_ticks[loc], time_ticks[loc])])
    try:
        if len(jumps) > 1:
            while (jumps[0][1][0] < 0) or (jumps[0][2][0] < 10):
                jumps.pop(0)
            while (jumps[-1][1][0] > 0) or (jumps[-1][2][0] > time_ticks[-1]-10):
                jumps.pop(-1)
        if len(jumps) < 2:
            # print "Cleaning jumps resulted in an error, so there is probably no"
            # print "well-detected event interval. Retruning empty list."
            return [], confidence_flag, 0.0
    except:
        # print "Cleaning jumps resulted in an error, so there is probably no"
        # print "well-detected event interval. Retruning empty list."
        return [], confidence_flag, 0.0

    jumps_all_clean, jumps_empty = test_jumps_cleanliness(jumps)
    n_cycles = 0
    while not jumps_all_clean:
        jumps = combine_jumps_together(jumps, time_ticks, smooth_vals)
        jumps_all_clean, jumps_empty = test_jumps_cleanliness(jumps)
        n_cycles += 1
        if (jumps_empty) or (n_cycles>100):
            break


    jumps = combine_jumps_together_smarter(jumps, time_ticks, smooth_vals)
    


    try:
        if len(jumps) > 1:
            while (jumps[0][1][0] < 0):
                jumps.pop(0)
            while (jumps[-1][1][0] > 0):
                jumps.pop(-1)
        if len(jumps) < 2:
            # print "Cleaning jumps resulted in an error, so there is probably no"
            # print "well-detected event interval. Retruning empty list."
            return [], confidence_flag, 0.0
    except:
        # print "Cleaning jumps resulted in an error, so there is probably no"
        # print "well-detected event interval. Retruning empty list."
        return [], confidence_flag, 0.0



    appliance_events = []
    abs_net_changes = []
    running_difference = np.array([])
    jump_histories = []
    jump_time_histories = []
    for jump in jumps:
        running_difference = running_difference + jump[1][0]
        running_difference = append(running_difference, jump[1][0])
        jump_histories.append([])
        for history in jump_histories:
            history.append(jump[1][0])
        jump_time_histories.append([])
        for time_history in jump_time_histories:
            time_history.append(jump[2])
        seq = range(len(running_difference))
        seq.reverse()
        for n in seq:
            net_change = running_difference[n]
            avg_height = (abs(jump_histories[n][0]) + abs(jump_histories[n][-1]))/2.0
            if (abs(net_change) < 50) and (abs(net_change) < 0.2*avg_height) and jump_histories[n][0] > 0 and jump_histories[n][-1] < 0:
                appliance_events.append(jump_time_histories[n][0][0])
                appliance_events.append(jump_time_histories[n][-1][-1])
                abs_net_changes.append(abs(net_change))
                confidence_flag = True
                break
    abs_net_changes = array(abs_net_changes)
    if len(appliance_events) > 2:
        ranked_appliance_events = []
        for n in range(len(abs_net_changes)):
            ranked_appliance_events.append([abs_net_changes[n], appliance_events[n*2], appliance_events[n*2+1]])
            ranked_appliance_events.sort()
        for candidate in ranked_appliance_events:
            trimmed_stream = vals[(time_ticks>(candidate[1]+3)) & (time_ticks<(candidate[2]-3))]        
            clipped_stream, event_height, event_height_std = mad_clipping(trimmed_stream, 3)
            before_stream = vals[(time_ticks>(candidate[1]-6)) & (time_ticks<(candidate[1]-3))]  
            after_stream = vals[(time_ticks>(candidate[2]+3)) & (time_ticks<(candidate[2]+6))]  
            event_height = event_height - ((median(before_stream) + median(after_stream))/2.)
            event_height_std = (event_height_std**2 + (before_stream.std()**2 + after_stream.std()**2))**0.5
            if (event_height > 0) and (event_height > 2*event_height_std):
                appliance_events = [candidate[1], candidate[2]]
                break
            else:
                continue
        

    if len(appliance_events) == 0:
        abs_net_changes = []
        running_difference = np.array([])
        jump_histories = []
        jump_time_histories = []
        for jump in jumps:
            running_difference = running_difference + jump[1][0]
            running_difference = append(running_difference, jump[1][0])
            jump_histories.append([])
            for history in jump_histories:
                history.append(jump[1][0])
            jump_time_histories.append([])
            for time_history in jump_time_histories:
                time_history.append(jump[2])
            seq = range(len(running_difference))
            seq.reverse()
            for n in seq:
                net_change = running_difference[n]
                avg_height = (abs(jump_histories[n][0]) + abs(jump_histories[n][-1]))/2.0
                if (abs(net_change) < 5) and jump_histories[n][0] > 0 and jump_histories[n][-1] < 0:
                    appliance_events.append(jump_time_histories[n][0][0])
                    appliance_events.append(jump_time_histories[n][-1][-1])
                    abs_net_changes.append(abs(net_change))
                    confidence_flag = True
                    break
    abs_net_changes = array(abs_net_changes)
    if len(appliance_events) > 2:
        ranked_appliance_events = []
        for n in range(len(abs_net_changes)):
            ranked_appliance_events.append([abs_net_changes[n], appliance_events[n*2], appliance_events[n*2+1]])
            ranked_appliance_events.sort()
        for candidate in ranked_appliance_events:
            trimmed_stream = vals[(time_ticks>(candidate[1]+3)) & (time_ticks<(candidate[2]-3))]        
            clipped_stream, event_height, event_height_std = mad_clipping(trimmed_stream, 3)
            before_stream = vals[(time_ticks>(candidate[1]-6)) & (time_ticks<(candidate[1]-3))]  
            after_stream = vals[(time_ticks>(candidate[2]+3)) & (time_ticks<(candidate[2]+6))]  
            event_height = event_height - ((median(before_stream) + median(after_stream))/2.)
            event_height_std = (event_height_std**2 + (before_stream.std()**2 + after_stream.std()**2))**0.5
            if (event_height > 0) and (event_height > 2*event_height_std):
                appliance_events = [candidate[1], candidate[2]]
                break
            else:
                continue
        
    
    # If the above fails but we're pretty sure something is going on because
    # the standard deviation of the signal is large, then just define the 
    # event to sandwich the bigger wiggles.
#     if len(appliance_events) == 0 and smooth_vals.std() > 3.0:
#         centroids, labels = kmeans2(smooth_vals, 2)
#         high_threshold_val = centroids.min() + smooth_vals.std()*2.0
#         trimmed_smooth_vals = smooth_vals[smooth_vals < high_threshold_val]
#         trimmed_time_ticks = time_ticks[smooth_vals < high_threshold_val]
#         trimmed_smooth_vals = trimmed_smooth_vals[10:-10]
#         trimmed_time_ticks = trimmed_time_ticks[10:-10]
#         if trimmed_smooth_vals.std() > 3.0:
#             centroids, labels = kmeans2(trimmed_smooth_vals, 2)
#             low_threshold_val = centroids.min() + trimmed_smooth_vals.std()
#             appliance_events = [ trimmed_time_ticks[where(trimmed_smooth_vals>low_threshold_val)[0].min()], trimmed_time_ticks[where(trimmed_smooth_vals>low_threshold_val)[0].max()] ]
#         else:
#             appliance_events = [ time_ticks[where(smooth_vals>high_threshold_val)[0].min()], time_ticks[where(smooth_vals>high_threshold_val)[0].max()] ]
    

    
    if len(appliance_events) > 0:
        trimmed_stream = vals[(time_ticks>(appliance_events[0]+3)) & (time_ticks<(appliance_events[1]-3))]        
        clipped_stream, event_height, event_height_std = mad_clipping(trimmed_stream, 3)
        before_stream = vals[(time_ticks>(appliance_events[0]-6)) & (time_ticks<(appliance_events[0]-3))]  
        after_stream = vals[(time_ticks>(appliance_events[1]+3)) & (time_ticks<(appliance_events[1]+6))]  
        event_height = event_height - ((median(before_stream) + median(after_stream))/2.)
        event_height_std = (event_height_std**2 + (before_stream.std()**2 + after_stream.std()**2))**0.5
        if (event_height < 0) or (event_height < 2*event_height_std):
            return [], False, 0.0
        else:
            return appliance_events, confidence_flag, event_height
    else:
            return appliance_events, confidence_flag, 0.0










house_dir = sys.argv[1]
tagged_training_filename = sys.argv[2]

# This is the directory we're working with presently
# house_dir = "data/H3/"

# This is the training file we're investigating
# tagged_training_filename = "Tagged_Training_07_30_1343631601.mat"

# Read in the matlab datafile
buf = io.loadmat(house_dir + tagged_training_filename)['Buffer']

# Parse the tagged info for appliance identification
taggingInfo_dict = parse_tagging_info(buf["TaggingInfo"][0][0])

# These are the voltage and current time-series FFT fundamental and first 5
# harmonics measurements.
LF1V = buf['LF1V'][0][0]                    # Nx6, N is number of time steps
LF1I = buf['LF1I'][0][0]                    # Nx6, N is number of time steps
LF2V = buf['LF2V'][0][0]                    # Nx6, N is number of time steps
LF2I = buf['LF2I'][0][0]                    # Nx6, N is number of time steps

# Time stamps, these are floats
L1_TimeTicks = buf['TimeTicks1'][0][0].flatten()
L2_TimeTicks = buf['TimeTicks2'][0][0].flatten()

# Spectrogram of high frequency noise. Each FFT vector is computed every 1.0667 secs
HF = buf['HF'][0][0]                        # 4096xN, N is number of FFT vectors
# Time stamps for the HF spectrogram, also floats
HF_TimeTicks = buf['TimeTicksHF'][0][0].flatten()

# Calculate power (by convolution)
L1_P = LF1V * LF1I.conjugate()              # Nx6, N is number of time steps
L2_P = LF2V * LF2I.conjugate()              # Nx6, N is number of time steps
#
# L1_ComplexPower = L1_P.sum(axis=1)          # length N, N is number of time steps
# L2_ComplexPower = L2_P.sum(axis=1)          # length N, N is number of time steps
#  
# # Extract components
# L1_Real = L1_ComplexPower.real
# L1_Imag = L1_ComplexPower.imag
# L1_Amp  = abs(L1_ComplexPower)
# L2_Real = L2_ComplexPower.real
# L2_Imag = L2_ComplexPower.imag
# L2_Amp  = abs(L2_ComplexPower)
# # Power Factor 
# L1_Pf = np.cos(np.angle(L1_P))
# L2_Pf = np.cos(np.angle(L2_P))





# for appliance_id in taggingInfo_dict:
#     taggingInfo_dict[appliance_id]["OnOffSeq_shifted"] = []
#     for interval in taggingInfo_dict[appliance_id]["OnOffSeq"]:
#         taggingInfo_dict[appliance_id]["OnOffSeq_shifted"].append(
#             (interval[0] - HF_TimeTicks[0], interval[1] - HF_TimeTicks[0]))





# Select a window in the HF Spectrogram time series to plot
# HF_start_index = 0
# HF_end_index = 100
# num_HF_timestamps = HF_end_index - HF_start_index

# This is the cropped HF time series
# HF_TimeTicks_window = HF_TimeTicks[HF_start_index:HF_end_index]

# key_array = np.array(taggingInfo_dict.keys())[where(np.array(taggingInfo_dict.keys())>20)[0]]
key_array = np.array(taggingInfo_dict.keys())

key_array.sort()

# sys.exit()

for appliance_id in key_array:
# for appliance_id in [10]:
    appliance_name = taggingInfo_dict[appliance_id]['ApplianceName'].replace("/", "")
    for interval in taggingInfo_dict[appliance_id]['OnOffSeq']:
    # for interval in [taggingInfo_dict[appliance_id]['OnOffSeq'][2]]:
        try:
            print appliance_id, appliance_name, interval
            start_time = interval[0] - 60
            end_time = interval[1] + 60
            write_to_error = False
            if interval[1] - interval[0] > 1000:
                error_log_file = file("error_log.txt", "a")
                error_log_file.write(house_dir[-3:-1] + "," + str(appliance_id) + "," + appliance_name.replace(" ", "") + "," + str(int(round((start_time + end_time)/2.0))) + "\n")
                error_log_file.close()
                continue

            a = HF_TimeTicks>=start_time
            b = HF_TimeTicks<=end_time
            HF_TimeTicks_indices = a*b
            HF_start_index = where(HF_TimeTicks_indices)[0].min()
            HF_end_index = where(HF_TimeTicks_indices)[0].max()+1
            HF_TimeTicks_window = HF_TimeTicks[HF_start_index:HF_end_index]

            num_HF_timestamps = HF_end_index - HF_start_index

            # Apply the time series window to the L1 data
            a = L1_TimeTicks>=HF_TimeTicks_window[0]
            b = L1_TimeTicks<=HF_TimeTicks_window[-1]
            L1_TimeTicks_indices = a*b
            L1_start_index = where(L1_TimeTicks_indices)[0].min()
            L1_end_index = where(L1_TimeTicks_indices)[0].max()+1
            L1_TimeTicks_window = L1_TimeTicks[L1_start_index:L1_end_index]
        
            L1_P_window = L1_P[L1_start_index:L1_end_index]

            # Apply the time series window to the L2 data
            a = L2_TimeTicks>=HF_TimeTicks_window[0]
            b = L2_TimeTicks<=HF_TimeTicks_window[-1]
            L2_TimeTicks_indices = a*b
            L2_start_index = where(L2_TimeTicks_indices)[0].min()
            L2_end_index = where(L2_TimeTicks_indices)[0].max()+1
            L2_TimeTicks_window = L2_TimeTicks[L2_start_index:L2_end_index]
        
            L2_P_window = L2_P[L2_start_index:L2_end_index]

            HF_TimeTicks_window_shifted = HF_TimeTicks_window - HF_TimeTicks_window[0]
            L1_TimeTicks_window_shifted = L1_TimeTicks_window - HF_TimeTicks_window[0]
            L2_TimeTicks_window_shifted = L2_TimeTicks_window - HF_TimeTicks_window[0]
            tagged_event_middle_time = (interval[0] + interval[1])/2.0 - HF_TimeTicks_window[0]
        
            L1_Amp_event_intervals, L1_Amp_confidence_flag, L1_Amp_height = event_detector2(L1_TimeTicks_window_shifted, abs(L1_P_window.sum(axis=1)))
            L2_Amp_event_intervals, L2_Amp_confidence_flag, L2_Amp_height = event_detector2(L2_TimeTicks_window_shifted, abs(L2_P_window.sum(axis=1)))
            
            
            if (L2_Amp_height == 0.0) and (L1_Amp_height == 0):
                print "No confident detection interval found, aborting."
                write_to_error = True
            
            elif L1_Amp_height > L2_Amp_height:
                detected_event_interval = array(L1_Amp_event_intervals)
                write_to_error = False
            elif L2_Amp_height > L1_Amp_height:
                detected_event_interval = array(L2_Amp_event_intervals)
                write_to_error = False
            

            if not write_to_error:
                L1_L2_plot_data = ((L1_TimeTicks_window_shifted, L1_P_window[:,0].real),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,1].real),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,2].real),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,3].real),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,4].real),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,5].real),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,0].imag),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,1].imag),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,2].imag),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,3].imag),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,4].imag),
                                   (L1_TimeTicks_window_shifted, L1_P_window[:,5].imag),
                                   (L1_TimeTicks_window_shifted, abs(L1_P_window[:,0])),
                                   (L1_TimeTicks_window_shifted, abs(L1_P_window[:,1])),
                                   (L1_TimeTicks_window_shifted, abs(L1_P_window[:,2])),
                                   (L1_TimeTicks_window_shifted, abs(L1_P_window[:,3])),
                                   (L1_TimeTicks_window_shifted, abs(L1_P_window[:,4])),
                                   (L1_TimeTicks_window_shifted, abs(L1_P_window[:,5])),
                                   (L1_TimeTicks_window_shifted, np.cos(np.angle(L1_P_window[:,0]))),
                                   (L1_TimeTicks_window_shifted, np.cos(np.angle(L1_P_window[:,1]))),
                                   (L1_TimeTicks_window_shifted, np.cos(np.angle(L1_P_window[:,2]))),
                                   (L1_TimeTicks_window_shifted, np.cos(np.angle(L1_P_window[:,3]))),
                                   (L1_TimeTicks_window_shifted, np.cos(np.angle(L1_P_window[:,4]))),
                                   (L1_TimeTicks_window_shifted, np.cos(np.angle(L1_P_window[:,5]))),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,0].real),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,1].real),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,2].real),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,3].real),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,4].real),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,5].real),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,0].imag),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,1].imag),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,2].imag),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,3].imag),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,4].imag),
                                   (L2_TimeTicks_window_shifted, L2_P_window[:,5].imag),
                                   (L2_TimeTicks_window_shifted, abs(L2_P_window[:,0])),
                                   (L2_TimeTicks_window_shifted, abs(L2_P_window[:,1])),
                                   (L2_TimeTicks_window_shifted, abs(L2_P_window[:,2])),
                                   (L2_TimeTicks_window_shifted, abs(L2_P_window[:,3])),
                                   (L2_TimeTicks_window_shifted, abs(L2_P_window[:,4])),
                                   (L2_TimeTicks_window_shifted, abs(L2_P_window[:,5])),
                                   (L2_TimeTicks_window_shifted, np.cos(np.angle(L2_P_window[:,0]))),
                                   (L2_TimeTicks_window_shifted, np.cos(np.angle(L2_P_window[:,1]))),
                                   (L2_TimeTicks_window_shifted, np.cos(np.angle(L2_P_window[:,2]))),
                                   (L2_TimeTicks_window_shifted, np.cos(np.angle(L2_P_window[:,3]))),
                                   (L2_TimeTicks_window_shifted, np.cos(np.angle(L2_P_window[:,4]))),
                                   (L2_TimeTicks_window_shifted, np.cos(np.angle(L2_P_window[:,5]))))

                L1_L2_plot_labels = ("L1_Real_0", 
                                     "L1_Real_1", 
                                     "L1_Real_2", 
                                     "L1_Real_3", 
                                     "L1_Real_4", 
                                     "L1_Real_5", 
                                     "L1_Imag_0", 
                                     "L1_Imag_1", 
                                     "L1_Imag_2", 
                                     "L1_Imag_3", 
                                     "L1_Imag_4", 
                                     "L1_Imag_5", 
                                     "L1_Amp_0", 
                                     "L1_Amp_1", 
                                     "L1_Amp_2", 
                                     "L1_Amp_3", 
                                     "L1_Amp_4", 
                                     "L1_Amp_5", 
                                     "L1_PF_0", 
                                     "L1_PF_1", 
                                     "L1_PF_2", 
                                     "L1_PF_3", 
                                     "L1_PF_4", 
                                     "L1_PF_5", 
                                     "L2_Real_0", 
                                     "L2_Real_1", 
                                     "L2_Real_2", 
                                     "L2_Real_3", 
                                     "L2_Real_4", 
                                     "L2_Real_5", 
                                     "L2_Imag_0", 
                                     "L2_Imag_1", 
                                     "L2_Imag_2", 
                                     "L2_Imag_3", 
                                     "L2_Imag_4", 
                                     "L2_Imag_5", 
                                     "L2_Amp_0", 
                                     "L2_Amp_1", 
                                     "L2_Amp_2", 
                                     "L2_Amp_3", 
                                     "L2_Amp_4", 
                                     "L2_Amp_5", 
                                     "L2_PF_0", 
                                     "L2_PF_1", 
                                     "L2_PF_2", 
                                     "L2_PF_3", 
                                     "L2_PF_4", 
                                     "L2_PF_5")


                before_event_time = detected_event_interval[0] - 5.0 # 5 second leeway
                after_event_time = detected_event_interval[1] + 5.0 # 5 second leeway
                middle_event_time = detected_event_interval.sum()/2


                before_event_HF_Time_index = where(abs(HF_TimeTicks_window_shifted - before_event_time) == min(abs(HF_TimeTicks_window_shifted - before_event_time)))[0] + HF_start_index
                middle_event_HF_Time_index = where(abs(HF_TimeTicks_window_shifted - middle_event_time) == min(abs(HF_TimeTicks_window_shifted - middle_event_time)))[0] + HF_start_index
                after_event_HF_Time_index = where(abs(HF_TimeTicks_window_shifted - after_event_time) == min(abs(HF_TimeTicks_window_shifted - after_event_time)))[0] + HF_start_index

                # Create the big plot

                fig = plt.figure(figsize=(40,21))
                ax0 = plt.subplot2grid((12,9), (0,0), rowspan=6)

                ax1 = plt.subplot2grid((12,9), (0, 1))
                ax2 = plt.subplot2grid((12,9), (1, 1))
                ax3 = plt.subplot2grid((12,9), (2, 1))
                ax4 = plt.subplot2grid((12,9), (3, 1))
                ax5 = plt.subplot2grid((12,9), (4, 1))
                ax6 = plt.subplot2grid((12,9), (5, 1))

                ax7 = plt.subplot2grid((12,9),  (0, 2))
                ax8 = plt.subplot2grid((12,9),  (1, 2))
                ax9 = plt.subplot2grid((12,9),  (2, 2))
                ax10 = plt.subplot2grid((12,9), (3, 2))
                ax11 = plt.subplot2grid((12,9), (4, 2))
                ax12 = plt.subplot2grid((12,9), (5, 2))

                ax13 = plt.subplot2grid((12,9), (0, 3))
                ax14 = plt.subplot2grid((12,9), (1, 3))
                ax15 = plt.subplot2grid((12,9), (2, 3))
                ax16 = plt.subplot2grid((12,9), (3, 3))
                ax17 = plt.subplot2grid((12,9), (4, 3))
                ax18 = plt.subplot2grid((12,9), (5, 3))

                ax19 = plt.subplot2grid((12,9), (0, 4))
                ax20 = plt.subplot2grid((12,9), (1, 4))
                ax21 = plt.subplot2grid((12,9), (2, 4))
                ax22 = plt.subplot2grid((12,9), (3, 4))
                ax23 = plt.subplot2grid((12,9), (4, 4))
                ax24 = plt.subplot2grid((12,9), (5, 4))

                ax25 = plt.subplot2grid((12,9),  (0, 5))
                ax26 = plt.subplot2grid((12,9),  (1, 5))
                ax27 = plt.subplot2grid((12,9),  (2, 5))
                ax28 = plt.subplot2grid((12,9), (3, 5))
                ax29 = plt.subplot2grid((12,9), (4, 5))
                ax30 = plt.subplot2grid((12,9), (5, 5))

                ax31 = plt.subplot2grid((12,9), (0, 6))
                ax32 = plt.subplot2grid((12,9), (1, 6))
                ax33 = plt.subplot2grid((12,9), (2, 6))
                ax34 = plt.subplot2grid((12,9), (3, 6))
                ax35 = plt.subplot2grid((12,9), (4, 6))
                ax36 = plt.subplot2grid((12,9), (5, 6))

                ax37 = plt.subplot2grid((12,9), (0, 7))
                ax38 = plt.subplot2grid((12,9), (1, 7))
                ax39 = plt.subplot2grid((12,9), (2, 7))
                ax40 = plt.subplot2grid((12,9), (3, 7))
                ax41 = plt.subplot2grid((12,9), (4, 7))
                ax42 = plt.subplot2grid((12,9), (5, 7))

                ax43 = plt.subplot2grid((12,9),  (0, 8))
                ax44 = plt.subplot2grid((12,9),  (1, 8))
                ax45 = plt.subplot2grid((12,9),  (2, 8))
                ax46 = plt.subplot2grid((12,9), (3, 8))
                ax47 = plt.subplot2grid((12,9), (4, 8))
                ax48 = plt.subplot2grid((12,9), (5, 8))

                ax49 = plt.subplot2grid((12,9), (6, 0), rowspan=2, colspan=9)
                ax50 = plt.subplot2grid((12,9), (8, 0), rowspan=2, colspan=9)
        
                ax51 = plt.subplot2grid((12,9), (10, 0), rowspan=2, colspan=9)

                axes=[ax1,  ax2,  ax3,  ax4,  ax5,  ax6,  
                      ax7,  ax8,  ax9,  ax10, ax11, ax12, 
                      ax13, ax14, ax15, ax16, ax17, ax18, 
                      ax19, ax20, ax21, ax22, ax23, ax24, 
                      ax25, ax26, ax27, ax28, ax29, ax30,
                      ax31, ax32, ax33, ax34, ax35, ax36, 
                      ax37, ax38, ax39, ax40, ax41, ax42,
                      ax43, ax44, ax45, ax46, ax47, ax48]


                plot_title = (house_dir[-3:-1] + " " + appliance_name + 
                              " from %.2f to %.2f seconds, datafile: " % (start_time, end_time) + 
                              tagged_training_filename)
                fig.suptitle(plot_title, fontsize=28, y=0.95)

                ax0.imshow(HF[:,HF_start_index:HF_end_index], origin="lower", interpolation="nearest",
                    extent=[HF_TimeTicks_window_shifted[0], HF_TimeTicks_window_shifted[-1], 0, 4096], 
                    aspect=3.3*num_HF_timestamps/4096.0)

                ax0.plot([before_event_time, before_event_time], [0, 4096], color="red")
                ax0.plot([middle_event_time, middle_event_time], [0, 4096], color="green")
                ax0.plot([after_event_time, after_event_time], [0, 4096], color="blue")

                ax0.set_xlim(HF_TimeTicks_window_shifted[0], HF_TimeTicks_window_shifted[-1])
                ax0.set_ylim(0, 4096)
                ax0.set_xlabel("Time From Start [s]")
                ax0.set_ylabel("FFT Vector (Frequency-space)")
                ax0.set_title("Spectrogram of\nHigh Frequency Noise")


                for n in range(len(axes)):
                    axes[n].plot(L1_L2_plot_data[n][0], L1_L2_plot_data[n][1], c="k", label=L1_L2_plot_labels[n])
                    axes[n].plot([before_event_time, before_event_time], [L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1], color="red")
                    axes[n].plot([middle_event_time, middle_event_time], [L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1], color="green")
                    axes[n].plot([after_event_time, after_event_time], [L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1], color="blue")
                    # axes[n].legend(loc="upper left")
                    # axes[n].set_ylabel(L1_L2_plot_labels[n])
                    axes[n].set_xlim(HF_TimeTicks_window_shifted[0], HF_TimeTicks_window_shifted[-1])
                    axes[n].set_ylim(L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1)
                    if n in [5, 11, 17, 23, 29, 35, 41, 47]:
                        axes[n].set_xlabel("Time From Start [s]")
                    else:
                        setp(axes[n].get_xticklabels(), visible=False)
                    if n in [0, 6, 12, 18, 24, 30, 36, 42]:
                        axes[n].set_title(L1_L2_plot_labels[n].replace("_", " ") + "-5")

                before_average_spectrum = HF[:,before_event_HF_Time_index-6:before_event_HF_Time_index].sum(axis=1) / float(HF[:,before_event_HF_Time_index-6:before_event_HF_Time_index].shape[1])

                event_average_spectrum = HF[:,middle_event_HF_Time_index-3:middle_event_HF_Time_index+3].sum(axis=1) / float(HF[:,middle_event_HF_Time_index-3:middle_event_HF_Time_index+3].shape[1])

                after_average_spectrum = HF[:,after_event_HF_Time_index-6:after_event_HF_Time_index].sum(axis=1) / float(HF[:,after_event_HF_Time_index-6:after_event_HF_Time_index].shape[1])

                diff_before_spectrum = event_average_spectrum - before_average_spectrum
                diff_after_spectrum = event_average_spectrum - after_average_spectrum

                #         smooth_off_average_spectrum = ndimage.filters.gaussian_filter1d(off_average_spectrum, 5)
                #         smooth_on_average_spectrum = ndimage.filters.gaussian_filter1d(on_average_spectrum, 5)
                #         smooth_diff_spectrum = smooth_on_average_spectrum - smooth_off_average_spectrum

                ax49.plot(before_average_spectrum, color="red", label="Before")
                ax49.plot(event_average_spectrum, color="green", label="During")
                ax49.plot(after_average_spectrum, color="blue", label="After")

                ax49.set_xlim(0,4096)
                ax49.set_ylim(0,255)
                # setp(ax49.get_xticklabels(), visible=False)
                ax49.legend(loc="upper left")
                ax49.set_ylabel("Spectra")
        
                ax50.plot(diff_before_spectrum, color="brown", label="During-Before")
                ax50.plot(diff_after_spectrum, color="darkcyan", label="During-After")

                ax50.legend(loc="upper left")
                ax50.set_xlim(0,4096)
                ax50.axhline()
                ax50.set_ylabel("Difference")
                ax50.set_xlabel("FFT Vector (Frequency-space)")
        
        
        
                context_L1_start_index = int(L1_start_index - ceil(600./0.1664888858795166)) # ten minute before
                context_L1_end_index = int(L1_end_index + ceil(600./0.1664888858795166)) # ten minute after
                context_L2_start_index = int(L2_start_index - ceil(600./0.1664888858795166)) # ten minute before
                context_L2_end_index = int(L2_end_index + ceil(600./0.1664888858795166)) # ten minute after
        
                ax51.plot(L1_TimeTicks[context_L1_start_index:context_L1_end_index]-HF_TimeTicks_window[0], abs(L1_P[context_L1_start_index:context_L1_end_index].sum(axis=1)), color="purple", label="L1 Amp")
                ax51.plot(L2_TimeTicks[context_L2_start_index:context_L2_end_index]-HF_TimeTicks_window[0], abs(L2_P[context_L2_start_index:context_L2_end_index].sum(axis=1)), color="orange", label="L2 Amp")
                ax51.legend(loc="upper left")
        
                first_time = min(min(L1_TimeTicks[context_L1_start_index:context_L1_end_index]-HF_TimeTicks_window[0]),         min(L2_TimeTicks[context_L2_start_index:context_L2_end_index]-HF_TimeTicks_window[0]))
                last_time = max(max(L1_TimeTicks[context_L1_start_index:context_L1_end_index]-HF_TimeTicks_window[0]), max(L2_TimeTicks[context_L2_start_index:context_L2_end_index]-HF_TimeTicks_window[0]))
        
                ax51.set_xlim(first_time, last_time)
        
                y_max_val = min(2000, max(max(abs(L1_P[context_L1_start_index:context_L1_end_index].sum(axis=1))), max(abs(L2_P[context_L2_start_index:context_L2_end_index].sum(axis=1)))))
                ax51.set_ylim(0, y_max_val)
                ax51.plot([before_event_time, before_event_time], [-100, y_max_val*1.1], color="red")
                ax51.plot([middle_event_time, middle_event_time], [-100, y_max_val*1.1], color="green")
                ax51.plot([after_event_time, after_event_time], [-100, y_max_val*1.1], color="blue")
                ax51.set_ylabel("Context Amp Plot")
        
        
                fig.subplots_adjust(hspace=0.175, wspace=0.25)

                canvas = FigureCanvas(fig)
                canvas.print_figure("plots/" + house_dir[-3:-1] + "_" + str(int(round((start_time + end_time)/2.0))) + "_" + appliance_name.replace(" ", "") + ".png", dpi=144, bbox_inches='tight')
                close("all")
    
                success_log_file = file("appliance_events.txt", "a")
                success_log_file.write(house_dir[-3:-1] + "," + str(appliance_id) + "," + appliance_name.replace(" ", "") + "," + str(HF_TimeTicks_window[0] + before_event_time) + "," + str(HF_TimeTicks_window[0] + after_event_time) + "\n")
                success_log_file.close()
            
            else:
                error_log_file = file("error_log.txt", "a")
                error_log_file.write(house_dir[-3:-1] + "," + str(appliance_id) + "," + appliance_name.replace(" ", "") + "," + str(int(round((start_time + end_time)/2.0))) + "\n")
                error_log_file.close()


        except:
            print "Failed due to crash."
            error_log_file = file("error_log.txt", "a")
            error_log_file.write(house_dir[-3:-1] + "," + str(appliance_id) + "," + appliance_name.replace(" ", "") + "," + str(int(round((start_time + end_time)/2.0))) + "\n")
            error_log_file.close()


