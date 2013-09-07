import ApplianceFeatures as af
import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
from scipy import ndimage
import sys, os
from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime, timedelta
import cPickle

# Calculate residual between appliance data series and transition


# testing_data_filename = "H4_09-18_testing.h5"
# house = "H1"

testing_data_filename = sys.argv[1]

house = testing_data_filename.split("_")[0]





def smooth_vals(vals):
    median_vals = ndimage.filters.median_filter(vals, 3) # median smooth with width 3 timeticks
    return median_vals
    
# Median Absolute Deviation clipping for input array of numbers.
def mad_clipping(input_data, sigma_clip_level):
    medval = np.median(input_data)
    sigma = 1.48 * np.median(abs(medval - input_data))
    high_sigma_clip_limit = medval + sigma_clip_level * sigma
    low_sigma_clip_limit = medval - sigma_clip_level * sigma
    clipped_data = input_data[(input_data>(low_sigma_clip_limit)) & (input_data<(high_sigma_clip_limit))]
    new_medval = np.median(clipped_data)
    new_sigma = 1.48 * np.median(abs(medval - clipped_data))
    return new_medval, new_sigma

def find_residual(transition_stream, appliance_stream, smooth_width):
    if smooth_width > 0:
        transition_vals = ndimage.filters.median_filter(transition_stream.values, smooth_width)
    else:
        transition_vals = transition_stream.values
    if transition_vals.min() > 0:
        min_transition_val, min_transition_val_std = mad_clipping(transition_vals[(transition_vals<transition_vals.min()*1.2)], 3)
    else:
        min_transition_val, min_transition_val_std = mad_clipping(transition_vals[(transition_vals<transition_vals.min()*0.8)], 3)
    if transition_vals.max()>0:
        max_transition_val, max_transition_val_std = mad_clipping(transition_vals[(transition_vals>transition_vals.max()*0.8)], 3)
    else:
        max_transition_val, max_transition_val_std = mad_clipping(transition_vals[(transition_vals>transition_vals.max()*1.2)], 3)
    transition_vals = transition_vals - min_transition_val

    if smooth_width > 0:
        appliance_vals = ndimage.filters.median_filter(appliance_stream.values, smooth_width)
    else:
        appliance_vals = appliance_stream.values
    if appliance_vals.min() > 0:
        min_appliance_val, min_appliance_val_std = mad_clipping(appliance_vals[(appliance_vals<appliance_vals.min()*1.2)], 3)
    else:
        min_appliance_val, min_appliance_val_std = mad_clipping(appliance_vals[(appliance_vals<appliance_vals.min()*0.8)], 3)
    if appliance_vals.max() > 0:
        max_appliance_val, max_appliance_val_std = mad_clipping(appliance_vals[(appliance_vals>appliance_vals.max()*0.8)], 3)
    else:
        max_appliance_val, max_appliance_val_std = mad_clipping(appliance_vals[(appliance_vals>appliance_vals.max()*1.2)], 3)
    appliance_vals = appliance_vals - min_appliance_val
    
    if not isnan(appliance_vals[0]) and not isnan(transition_vals[0]):
        appliance_stream_length = len(appliance_vals)
        length_diff = len(transition_vals) - appliance_stream_length
        residuals = []
        max_differences = []
        for n in range(length_diff+1):
            residual = sum(abs(transition_vals[n:appliance_stream_length+n] - appliance_vals))/appliance_stream_length
            max_difference = (abs(transition_vals[n:appliance_stream_length+n] - appliance_vals)).max()
            residuals.append(residual)
            max_differences.append(max_difference)
        residuals = np.array(residuals)
        max_differences = np.array(max_differences)
        if abs(abs(max_appliance_val) - abs(min_appliance_val)) < 2.0:
            return None, None, None
        else:
            return residuals.min()/(max_appliance_val-min_appliance_val), max_differences[np.where(residuals==residuals.min())[0][0]]/(max_appliance_val-min_appliance_val),  np.where(residuals==residuals.min())[0][0]
    else:
        return None, None, None

def find_residual_robust(transition_stream, appliance_stream):
    smooth_width_array = [0, 3, 6, 12]
    residual_list = []
    for smooth_width in smooth_width_array:
        residual, max_diff, shift,  = find_residual(transition_stream, appliance_stream, smooth_width)
        if residual:
            if not isnan(residual):
                residual_list.append([residual, max_diff, shift])
    residual_list.sort()
    if len(residual_list)>0:
        return residual_list[0][0], residual_list[0][2]
    else:
        return None, None


# file_list = os.listdir("data/hdf5storage/")
# for testing_data_filename in file_list:
#     if testing_data_filename[-3:] == ".h5" and testing_data_filename.split("_")[0]==house:




transition_filename = "all_transitions.csv"
transition_data = np.loadtxt(transition_filename, skiprows=1, delimiter=",",
    dtype=[ ("timestamp",np.float64),
            ("L1_real",np.float32), 
            ("L1_imag",np.float32), 
            ("L2_real",np.float32),
            ("L2_imag",np.float32) ] )

# house,id,name,transition,L1_real,L1_imag,L2_real,L2_imag
simple_features = np.loadtxt("simple_training_features.csv", skiprows=1, delimiter=",",
    dtype=[ ("house","S2"), ("id", int), ("name", "S27"), ("transition", "S3"),
            ("L1_real",np.float32), 
            ("L1_imag",np.float32), 
            ("L2_real",np.float32),
            ("L2_imag",np.float32)])
simple_features = simple_features[simple_features["house"]==house]

# house,id,name,start,stop
appliance_training_data = np.loadtxt("data/eventtimes.csv", skiprows=1, delimiter=",",
    dtype=[ ("house","S2"), ("id", int), ("name", "S27"),
        ("start_time",np.float64), 
        ("stop_time",np.float64)])

appliance_training_nums = np.where(appliance_training_data["house"]==house)[0]
appliance_training_data = appliance_training_data[appliance_training_nums]
appliance_training_data = append_fields(appliance_training_data, "appnum", appliance_training_nums, usemask=False)


data_stream = af.ElectricTimeStream("data/hdf5storage/" + testing_data_filename)
# Extract Components on the ap object
data_stream.ExtractComponents()



stream_start_time = min((data_stream.l1.index[0] - datetime(1970, 1, 1)).total_seconds(),
                        (data_stream.l2.index[0] - datetime(1970, 1, 1)).total_seconds())
stream_end_time = max((data_stream.l1.index[-1] - datetime(1970, 1, 1)).total_seconds(),
                        (data_stream.l2.index[-1] - datetime(1970, 1, 1)).total_seconds())

# Use this to hard-code a specific testing temporal region
# stream_start_time = (datetime(2012, 7, 18, 7, 02, 30) - datetime(1970, 1, 1)).total_seconds()
# stream_end_time = (datetime(2012, 7, 18, 7, 5, 00) - datetime(1970, 1, 1)).total_seconds()

transition_filename = "all_transitions.csv"
transition_data = np.loadtxt(transition_filename, skiprows=1, delimiter=",",
    dtype=[ ("timestamp",np.float64),
            ("L1_real",np.float32), 
            ("L1_imag",np.float32), 
            ("L2_real",np.float32),
            ("L2_imag",np.float32) ] )
transition_data = transition_data[(transition_data["timestamp"]>stream_start_time)&(transition_data["timestamp"]<stream_end_time)]

stream_L1_real = data_stream.l1.real.sum(axis=1)
stream_L1_imag = data_stream.l1.imag.sum(axis=1)
stream_L2_real = data_stream.l2.real.sum(axis=1)
stream_L2_imag = data_stream.l2.imag.sum(axis=1)


# Read all the comparison appliances into memory, store in a dictionary
comparison_appliance_dictionary = {}
for appliance_num in appliance_training_nums:
    comparison_appliance_dictionary[appliance_num] = {}
    
    comparison_appliance_training_data = appliance_training_data[appliance_training_data["appnum"]==appliance_num]
    

    comparison_appliance = af.Appliance(appliance_num)
    comparison_appliance.ExtractComponents()
    
    for transition_type in ["on", "off"]:
        if transition_type == "on":
            comparison_appliance_time = datetime(1970, 1, 1) + timedelta(seconds=comparison_appliance_training_data["start_time"][0]+5)
        if transition_type == "off":
            comparison_appliance_time = datetime(1970, 1, 1) + timedelta(seconds=comparison_appliance_training_data["stop_time"][0]-5)                

        comparison_appliance_L1_stream = comparison_appliance.l1[ 
            (comparison_appliance.l1.index>comparison_appliance_time-timedelta(seconds=10)) & 
            (comparison_appliance.l1.index<comparison_appliance_time+timedelta(seconds=10))]
        comparison_appliance_L2_stream = comparison_appliance.l2[ 
            (comparison_appliance.l2.index>comparison_appliance_time-timedelta(seconds=10)) & 
            (comparison_appliance.l2.index<comparison_appliance_time+timedelta(seconds=10))]

        comparison_appliance_L1_real = comparison_appliance_L1_stream.real.sum(axis=1)
        comparison_appliance_L1_imag = comparison_appliance_L1_stream.imag.sum(axis=1)
        comparison_appliance_L2_real = comparison_appliance_L2_stream.real.sum(axis=1)
        comparison_appliance_L2_imag = comparison_appliance_L2_stream.imag.sum(axis=1)

        comparison_appliance_dictionary[appliance_num][transition_type] = [comparison_appliance_L1_real, comparison_appliance_L1_imag, comparison_appliance_L2_real, comparison_appliance_L2_imag]





classified_transitions = []
for transition in transition_data:
    comparison_results = []
    for appliance_featureset in simple_features:
        L1_rough_real_residual = abs(transition["L1_real"] - appliance_featureset["L1_real"])
        L1_rough_imag_residual = abs(transition["L1_imag"] - appliance_featureset["L1_imag"])
        L2_rough_real_residual = abs(transition["L2_real"] - appliance_featureset["L2_real"])
        L2_rough_imag_residual = abs(transition["L2_imag"] - appliance_featureset["L2_imag"])
        max_power_val = max(abs(transition["L1_real"]), abs(transition["L1_imag"]), abs(transition["L2_real"]), abs(transition["L2_imag"]))
        residual_threshold = max(max_power_val*0.3, 30)
        if (max(L1_rough_real_residual, L1_rough_imag_residual, L2_rough_real_residual, L2_rough_imag_residual)<residual_threshold):
            transition_type = appliance_featureset["transition"]
            comparison_appliance_nums = appliance_training_nums[appliance_training_data["id"]==appliance_featureset["id"]]
            transition_time = datetime(1970, 1, 1) + timedelta(seconds=transition["timestamp"])
            transition_L1_real = stream_L1_real[(stream_L1_real.index > transition_time - timedelta(seconds=25)) & 
                                                (stream_L1_real.index < transition_time + timedelta(seconds=25))]
            transition_L1_imag = stream_L1_imag[(stream_L1_imag.index > transition_time - timedelta(seconds=25)) & 
                                                (stream_L1_imag.index < transition_time + timedelta(seconds=25))]
            transition_L2_real = stream_L2_real[(stream_L2_real.index > transition_time - timedelta(seconds=25)) & 
                                                (stream_L2_real.index < transition_time + timedelta(seconds=25))]
            transition_L2_imag = stream_L2_imag[(stream_L2_imag.index > transition_time - timedelta(seconds=25)) & 
                                                (stream_L2_imag.index < transition_time + timedelta(seconds=25))]
                                        
            for appliance_num in comparison_appliance_nums:
                comparison_appliance_training_data = appliance_training_data[appliance_training_data["appnum"]==appliance_num]
#                 if transition_type == "on":
#                     comparison_appliance_time = datetime(1970, 1, 1) + timedelta(seconds=comparison_appliance_training_data["start_time"][0]+5)
#                 if transition_type == "off":
#                     comparison_appliance_time = datetime(1970, 1, 1) + timedelta(seconds=comparison_appliance_training_data["stop_time"][0]-5)                
#         
#                 comparison_appliance = af.Appliance(appliance_num)
#                 comparison_appliance.ExtractComponents()
#                 comparison_appliance_L1_stream = comparison_appliance.l1[ 
#                     (comparison_appliance.l1.index>comparison_appliance_time-timedelta(seconds=10)) & 
#                     (comparison_appliance.l1.index<comparison_appliance_time+timedelta(seconds=10))]
#                 comparison_appliance_L2_stream = comparison_appliance.l2[ 
#                     (comparison_appliance.l2.index>comparison_appliance_time-timedelta(seconds=10)) & 
#                     (comparison_appliance.l2.index<comparison_appliance_time+timedelta(seconds=10))]
#         
#                 comparison_appliance_L1_real = comparison_appliance_L1_stream.real.sum(axis=1)
#                 comparison_appliance_L1_imag = comparison_appliance_L1_stream.imag.sum(axis=1)
#                 comparison_appliance_L2_real = comparison_appliance_L2_stream.real.sum(axis=1)
#                 comparison_appliance_L2_imag = comparison_appliance_L2_stream.imag.sum(axis=1)
                
                comparison_appliance_L1_real = comparison_appliance_dictionary[appliance_num][transition_type][0]
                comparison_appliance_L1_imag = comparison_appliance_dictionary[appliance_num][transition_type][1]
                comparison_appliance_L2_real = comparison_appliance_dictionary[appliance_num][transition_type][2]
                comparison_appliance_L2_imag = comparison_appliance_dictionary[appliance_num][transition_type][3]
        
                num_accepted_residuals = 0
                sum_residual = 0
                image_shifts_list = []
                if abs(appliance_featureset["L1_real"]) > 10:
                    L1_real_residual, L1_real_shift = find_residual_robust(transition_L1_real, comparison_appliance_L1_real)
                    if L1_real_residual and not isnan(L1_real_residual):
                        num_accepted_residuals += 1
                        sum_residual += L1_real_residual
                        image_shifts_list.append(L1_real_shift)
                else:
                    L1_real_residual = None
        
                if abs(appliance_featureset["L1_imag"]) > 10:
                    L1_imag_residual, L1_imag_shift = find_residual_robust(transition_L1_imag, comparison_appliance_L1_imag)
                    if L1_imag_residual and not isnan(L1_imag_residual):
                        num_accepted_residuals += 1
                        sum_residual += L1_imag_residual
                        image_shifts_list.append(L1_imag_shift)
                else:
                    L1_imag_residual = None
        
                if abs(appliance_featureset["L2_real"]) > 10:
                    L2_real_residual, L2_real_shift = find_residual_robust(transition_L2_real, comparison_appliance_L2_real)
                    if L2_real_residual and not isnan(L2_real_residual):
                        num_accepted_residuals += 1
                        sum_residual += L2_real_residual
                        image_shifts_list.append(L2_real_shift)
                else:
                    L2_real_residual = None
        
                if abs(appliance_featureset["L2_imag"]) > 10:
                    L2_imag_residual, L2_imag_shift = find_residual_robust(transition_L2_imag, comparison_appliance_L2_imag)
                    if L2_imag_residual and not isnan(L2_imag_residual):
                        num_accepted_residuals += 1
                        sum_residual += L2_imag_residual
                        image_shifts_list.append(L2_imag_shift)
                else:
                    L2_imag_residual = None
                if num_accepted_residuals > 0:
                    average_residual = sum_residual/num_accepted_residuals                
                    comparison_results.append([average_residual, comparison_appliance_training_data["id"], comparison_appliance_training_data["name"], transition_type, image_shifts_list])
                                        
    # If the comparison_results list contains multiple entries for the same appliance, take the best
    comparison_results.sort()
    accepted_candidate_ids = []
    accepted_candidates = []
    for candidate in comparison_results:
        if candidate[1] not in accepted_candidate_ids and not isnan(candidate[0]):
            if candidate[0] < 0.1: # Enforce threshold on residual metric. 
                # Units are fractional average power difference per timetick of appliance comparison stream
                accepted_candidates.append([candidate[1][0], candidate[2][0], candidate[3], candidate[0], candidate[4]])
                accepted_candidate_ids.append(candidate[1])
    if len(accepted_candidates)>0:
        new_transition = [transition["timestamp"], accepted_candidates]
        classified_transitions.append(new_transition)


    

        
    

L1_Amp = data_stream.l1.amp.sum(axis=1)
L2_Amp = data_stream.l2.amp.sum(axis=1)

L1_time_length = (L1_Amp.index[-1] - L1_Amp.index[0]).total_seconds()
L2_time_length = (L2_Amp.index[-1] - L2_Amp.index[0]).total_seconds()

n_plot_rows_L1 = np.ceil(L1_time_length/1800.)
n_plot_rows_L2 = np.ceil(L2_time_length/1800.)
n_plot_rows = int(max(n_plot_rows_L1, n_plot_rows_L2))

plot_height=5000
fig, axes = plt.subplots(n_plot_rows, 1, figsize=(40,n_plot_rows*5))
for n in range(n_plot_rows):
    L1_time_start = L1_Amp.index[0] + timedelta(seconds=1800*n)
    L1_time_end = L1_Amp.index[0] + timedelta(seconds=1800*(n+1))
    L1_amp_trimmed = L1_Amp[(L1_Amp.index>L1_time_start)&(L1_Amp.index<L1_time_end)]
    axes[n].plot(L1_amp_trimmed.index, L1_amp_trimmed.values, c="purple", lw=2)

    L2_time_start = L2_Amp.index[0] + timedelta(seconds=1800*n)
    L2_time_end = L2_Amp.index[0] + timedelta(seconds=1800*(n+1))
    L2_amp_trimmed = L2_Amp[(L2_Amp.index>L2_time_start)&(L2_Amp.index<L2_time_end)]
    axes[n].plot(L2_amp_trimmed.index, L2_amp_trimmed.values, c="orange", lw=2)




    plot_start_time = min(L2_time_start, L1_time_start)
    plot_end_time = max(L2_time_end, L1_time_end)

    plot_start_time_seconds = (plot_start_time - datetime(1970,1,1)).total_seconds()
    plot_end_time_seconds = (plot_end_time - datetime(1970,1,1)).total_seconds()

    for transition in classified_transitions:
        transition_timestamp = datetime(1970,1,1) + timedelta(seconds=transition[0])
        if transition_timestamp>plot_start_time and transition_timestamp<plot_end_time:
            desc_text = transition[1][0][1] + "-" + transition[1][0][2] + "-" + str(round(transition[1][0][3],3))
            if len(transition[1])>1:
                desc_text = desc_text + "+" + str(len(transition[1])-1)
            transition_type = transition[1][0][2]
            if transition_type == "on":
                transition_line_color = "green"
            else:
                transition_line_color = "red"

            axes[n].plot([transition_timestamp, transition_timestamp], [0, plot_height], color=transition_line_color, lw=2)
            if transition_line_color == "green":
                axes[n].text(transition_timestamp + timedelta(seconds=2), plot_height-100, desc_text, va="top", ha="left", color="k", rotation=90)
                axes[n].scatter([transition_timestamp], [plot_height-50], marker="^", color="green", s=80)
            else:
                axes[n].text(transition_timestamp - timedelta(seconds=2), plot_height-100, desc_text, va="top", ha="right", color="k", rotation=90)
                axes[n].scatter([transition_timestamp], [plot_height-50], marker="v", color="red", s=80)
                
    

    axes[n].set_xlim(plot_start_time, plot_end_time)
    axes[n].set_ylim(0,plot_height)
    axes[n].set_ylabel("Line Amplitudes")
    axes[n].set_xlabel("Timestamp")

canvas = FigureCanvas(fig)
canvas.print_figure("classified_plots/" + testing_data_filename.split(".")[0] + ".png", dpi=72, bbox_inches='tight')
close("all")



with open("classified_plots/" + testing_data_filename.split(".")[0] + "_classified.pkl",'w+b') as f:
    cPickle.dump(classified_transitions, f)





# with open("classified_plots/" + testing_data_filename.split(".")[0] + "_classified.pkl",'rb') as fp:
#     ct=cPickle.load(fp)