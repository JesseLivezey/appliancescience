import numpy as np
import pandas as pd
import os
from operator import itemgetter
import ApplianceFeatures as ap

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime, timedelta

training_features_filename = "simple_training_features.csv"
# house,id,name,transition,L1_real,L1_imag,L2_real,L2_imag
training_features = np.loadtxt(training_features_filename, delimiter=",", skiprows=1,
    dtype=[ ("house","S2"), ("id", int), ("name", "S27"), ("transition", "S3"),
            ("L1_real",np.float32), 
            ("L1_imag",np.float32), 
            ("L2_real",np.float32),
            ("L2_imag",np.float32)])




file_list = os.listdir("data/hdf5storage/")

# filelist goes through 31
datafilename = file_list[30]    # for plotting





testing_features_filename = "simple_features_" + datafilename[:-3] + ".csv"
house = datafilename[:2]



# timestamp,L1_real,L1_imag,L2_real,L2_imag
testing_features = np.loadtxt(testing_features_filename, delimiter=",", skiprows=1,
    dtype=[ ("timestamp", np.float64),
            ("L1_real",np.float32), 
            ("L1_imag",np.float32),
            ("L2_real",np.float32),
            ("L2_imag",np.float32)])

# Reduce the total training features list to only those in the house
# corresponding to the testing filename we're interested in
training_features = training_features[training_features["house"] == house]

# Combine multiple instances of a training example into one generic feature.
# Intelligently exclude outliers
training_id_list = list(set(training_features["id"]))
for id_num in training_id_list:
    # First do it for "on" transitions
    training_features_subset = training_features[(training_features["id"]==id_num) & (training_features["transition"]=="on")]
    if len(training_features_subset) > 2:
        new_feature = [ training_features_subset[0]["house"],
                        training_features_subset[0]["id"],
                        training_features_subset[0]["name"],
                        training_features_subset[0]["transition"],
                        median(training_features_subset["L1_real"]),
                        median(training_features_subset["L1_imag"]),
                        median(training_features_subset["L2_real"]),
                        median(training_features_subset["L2_imag"])]
                                

                                
                                



big_candidate_transitions_list = []

for testf in testing_features:
    candidate_transitions = []
    seconds_timestamp = testf["timestamp"]
    timestamp = datetime(1970,1,1) + timedelta(seconds=seconds_timestamp)
    for trainf in training_features:
        L1_real_residual = abs(testf["L1_real"] - trainf["L1_real"])
        L1_imag_residual = abs(testf["L1_imag"] - trainf["L1_imag"])
        L2_real_residual = abs(testf["L2_real"] - trainf["L2_real"])
        L2_imag_residual = abs(testf["L2_imag"] - trainf["L2_imag"])
        overall_residual = L1_real_residual + L1_imag_residual + L2_real_residual + L2_imag_residual
        if (overall_residual<15):
            candidate_transitions.append([overall_residual, trainf["id"], trainf["name"], trainf["transition"], seconds_timestamp, timestamp])
    if candidate_transitions:
        candidate_transitions.sort()
        vetted_candidate_transitions = [candidate_transitions[0]]
        for ct in candidate_transitions:
            dup_id = False
            for vt in vetted_candidate_transitions:
                if (ct[2] == vt[2]) and (ct[4] == vt[4]):
                    dup_id = True
            if not dup_id:
                vetted_candidate_transitions.append(ct)
        big_candidate_transitions_list.append(vetted_candidate_transitions)

on_seconds_timestamps = []
off_seconds_timestamps = []
candidate_on_transitions = []
candidate_off_transitions = []
for candidate_list in big_candidate_transitions_list:
    candidate_intervals = []    # This is where we'll record possible intervals
    for candidate in candidate_list:
        if candidate[3]=="on":
            candidate_on_transitions.append(candidate)
            on_seconds_timestamps.append(candidate[4])
        elif candidate[3]=="off":
            candidate_off_transitions.append(candidate)
            off_seconds_timestamps.append(candidate[4])

on_seconds_timestamps = sorted(list(set(on_seconds_timestamps)))
off_seconds_timestamps = sorted(list(set(off_seconds_timestamps)))


all_candidate_intervals = []
for on_candidate in candidate_on_transitions:
    on_residual = on_candidate[0]
    on_id = on_candidate[1]
    on_name = on_candidate[2]
    on_seconds_timestamp = on_candidate[4]
    on_timestamp = on_candidate[5]
    candidate_intervals = []
    for off_candidate in candidate_off_transitions:
        if off_candidate[1] == on_id and off_candidate[4]>on_seconds_timestamp and (off_candidate[4] - on_seconds_timestamp)<3600*2:
            interval_residual = on_residual + off_candidate[0]
            candidate_intervals.append([off_candidate[4] - on_seconds_timestamp, interval_residual, on_id, on_name, on_seconds_timestamp, off_candidate[4], on_timestamp, off_candidate[5]])
    candidate_intervals.sort()
    if len(candidate_intervals)>0:
        all_candidate_intervals.append(candidate_intervals[0])

# duration, interval_residual, id, name, on_seconds_timestamp, off_seconds_timestamp, on_timestamp, off_timestamp
# 0         1                  2   3     4                     5                      6             7
all_candidate_intervals_sorted = sorted(all_candidate_intervals, key=itemgetter(6)) # sorted by on_timestamp (increasing)

final_intervals = []
for on_seconds_timestamp in on_seconds_timestamps:
    candidate_intervals_same_start_time = []
    for candidate_interval in all_candidate_intervals_sorted:
        if on_seconds_timestamp == candidate_interval[4]:
            candidate_intervals_same_start_time.append(candidate_interval)
    if len(candidate_intervals_same_start_time)>0:
        duration_array = np.array(candidate_intervals_same_start_time)[:,0].astype(float)
        min_duration = duration_array.min()
        if min_duration > 15: # require an appliance to be on for more than 15 seconds (shorter => probably wrong)
            # If only one candidate interval has the minimum duration, that's it
            # If multiple candidate intervals have the same minimum duration, no clear correct identification
            candidate_intervals_min_duration = []
            for candidate_interval in candidate_intervals_same_start_time:
                if candidate_interval[0] == min_duration and candidate_interval[1]<20:
                    candidate_intervals_min_duration.append(candidate_interval)
            if len(candidate_intervals_min_duration) == 1:
                final_intervals.append(candidate_intervals_min_duration[0])

# Remove multiple instances of a single appliance being on simultaneously
active_appliances = []
for interval in final_intervals:
    active_appliances.append(interval[2])
active_appliances = sorted(list(set(active_appliances)))

clean_final_intervals = []
for active_appliance_id in active_appliances:
    same_appliance_intervals = []
    for interval in final_intervals:
        if interval[2] == active_appliance_id:
            same_appliance_intervals.append(interval)
    for test_interval in same_appliance_intervals:
        test_start_time = test_interval[4]
        test_end_time = test_interval[5]
        overlaps = 0
        for match_interval in same_appliance_intervals:
            match_start_time = match_interval[4]
            match_end_time = match_interval[5]
            if (match_start_time < test_end_time and match_start_time > test_start_time) or (match_end_time < test_end_time and match_end_time > test_start_time):
                overlaps += 1
        if overlaps == 0:
            clean_final_intervals.append(test_interval)
    
    
    
output_file = file("classified_series/" + datafilename.split(".")[0] + "_classified_intervals.txt", "w")
for interval in clean_final_intervals:
    output_file.write(str(interval[4]) + "," + str(interval[5]) + "," + house + "," + str(interval[2]) + "\n")
output_file.close()



a = ap.ElectricTimeStream("data/hdf5storage/" + datafilename)
# Extract Components on the ap object
a.ExtractComponents()

# Trim to specified time range. Can take date-time object or strings
# a_trimmed = a.l1.amp["0"][(a.l1.index>"2012-04-13 22:04:51.253961") &(a.l1.index<"2012-04-13 22:05:51.253961")]

L1_Amp = a.l1.amp.sum(axis=1)
L2_Amp = a.l2.amp.sum(axis=1)

L1_time_length = (L1_Amp.index[-1] - L1_Amp.index[0]).total_seconds()
L2_time_length = (L2_Amp.index[-1] - L2_Amp.index[0]).total_seconds()

n_plot_rows_L1 = np.ceil(L1_time_length/1800.)
n_plot_rows_L2 = np.ceil(L2_time_length/1800.)
n_plot_rows = int(max(n_plot_rows_L1, n_plot_rows_L2))

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

    # amp_plot_limit = min(4000, max(max(L1_amp_trimmed.values)*1.1, max(L2_amp_trimmed.values)*1.1))
    amp_plot_limit = 4000
    axes[n].set_ylim(0, amp_plot_limit)
    plot_start_time = min(L2_time_start, L1_time_start)
    plot_end_time = max(L2_time_end, L1_time_end)
    axes[n].set_xlim(min(L2_time_start, L1_time_start), max(L2_time_end, L1_time_end))

    for classified_event in clean_final_intervals:
        if classified_event[6] > plot_start_time and classified_event[6] < plot_end_time:
            axes[n].plot([classified_event[6], classified_event[6]], [0, amp_plot_limit], color="green", lw=2)
            axes[n].text(classified_event[6], uniform(0.5*amp_plot_limit, 0.85*amp_plot_limit), classified_event[3], va="bottom", ha="left", color="green")
        if classified_event[7] > plot_start_time and classified_event[7] < plot_end_time:
            axes[n].plot([classified_event[7], classified_event[7]], [0, amp_plot_limit], color="red", lw=2)
            axes[n].text(classified_event[7], uniform(0.15*amp_plot_limit, 0.5*amp_plot_limit), classified_event[3], va="bottom", ha="right", color="red")
    

    axes[n].set_ylabel("Line Amplitudes")
    axes[n].set_xlabel("Timestamp")

canvas = FigureCanvas(fig)
canvas.print_figure("classified_series/" + datafilename.split(".")[0] + "_classified.png", dpi=72, bbox_inches='tight')
close("all")