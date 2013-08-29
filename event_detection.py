import numpy as np
from matplotlib.pylab import *
from scipy import io
from cmath import phase
import sys

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy import ndimage, signal

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

def measure_jump(vals, loc, search_width=1.0):
    """
    Find the difference when going from one state to another.
    search_width is in seconds.
    """
    search_width_ticks = ceil(search_width/(time_ticks[1]-time_ticks[0]))
    previous_plateau = vals[loc - 2 - search_width_ticks:loc - 2]
    after_plateau = vals[loc + 2:loc + 2 + search_width_ticks]
    previous_mean = previous_plateau.mean()
    previous_std = previous_plateau.std()
    after_mean = after_plateau.mean()
    after_std = after_plateau.std()
    jump = after_mean - previous_mean
    jump_std = (after_std**2 + previous_std**2)**0.5
    return jump, jump_std

def event_detector2(time_ticks, vals):
    # smooth with a width of 1 second
    smooth_vals = ndimage.filters.gaussian_filter1d(vals, 1.0/time_ticks[1]-time_ticks[0])
    # take the gradient to find where the changes occur
    smooth_detection_stream = np.gradient(smooth_vals)

    peak_locs = signal.find_peaks_cwt(abs(smooth_detection_stream), np.array([1]), min_snr=1)
    half_amp = (smooth_detection_stream.max() - smooth_detection_stream.min())/2.0
    vetted_peak_locs = list(where(abs(smooth_detection_stream[peak_locs]) > half_amp/5.0)[0])
    peak_indices = list(array(peak_locs)[vetted_peak_locs])
    peak_times = time_ticks[peak_indices]
    peak_data = smooth_detection_stream[peak_indices]

    jumps = []
    for loc in peak_indices:
        jump_height, jump_std = measure_jump(vals, loc, search_width=1.0)
        if abs(jump_height) > 3*jump_std:
            jumps.append([loc, (jump_height, jump_std)])
    if len(jumps) > 1:
        while (jumps[0][1][0] < 0):
            jumps.pop(0)
        while (jumps[-1][1][0] > 0):
            jumps.pop(-1)

        detected_interval = (time_ticks[jumps[0][0]], time_ticks[jumps[-1][0]])
        return detected_interval
    else:
        return ()









# house_dir = sys.argv[1]
# tagged_training_filename = sys.argv[2]

# This is the directory we're working with presently
house_dir = "data/H2/"

# This is the training file we're investigating
tagged_training_filename = "Tagged_Training_06_13_1339570801.mat"


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
L1_ComplexPower = L1_P.sum(axis=1)          # length N, N is number of time steps
L2_ComplexPower = L2_P.sum(axis=1)          # length N, N is number of time steps
 
# Extract components
L1_Real = L1_ComplexPower.real
L1_Imag = L1_ComplexPower.imag
L1_Amp  = abs(L1_ComplexPower)
L2_Real = L2_ComplexPower.real
L2_Imag = L2_ComplexPower.imag
L2_Amp  = abs(L2_ComplexPower)
# Power Factor 
L1_Pf = np.cos(np.angle(L1_P))
L2_Pf = np.cos(np.angle(L2_P))





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

# for appliance_id in taggingInfo_dict.keys():
for appliance_id in [35]:
    appliance_name = taggingInfo_dict[appliance_id]['ApplianceName']
    for interval in taggingInfo_dict[appliance_id]['OnOffSeq']:
    # for interval in [taggingInfo_dict[appliance_id]['OnOffSeq'][3]]:

        start_time = interval[0] - 75
        end_time = interval[1] + 75

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

        L1_Real_window = L1_Real[L1_start_index:L1_end_index]
        L1_Imag_window = L1_Imag[L1_start_index:L1_end_index]
        L1_Amp_window = L1_Amp[L1_start_index:L1_end_index]
        L1_Pf_window = L1_Pf[L1_start_index:L1_end_index]

        # Apply the time series window to the L2 data
        a = L2_TimeTicks>=HF_TimeTicks_window[0]
        b = L2_TimeTicks<=HF_TimeTicks_window[-1]
        L2_TimeTicks_indices = a*b
        L2_start_index = where(L2_TimeTicks_indices)[0].min()
        L2_end_index = where(L2_TimeTicks_indices)[0].max()+1
        L2_TimeTicks_window = L2_TimeTicks[L2_start_index:L2_end_index]

        L2_Real_window = L2_Real[L2_start_index:L2_end_index]
        L2_Imag_window = L2_Imag[L2_start_index:L2_end_index]
        L2_Amp_window = L2_Amp[L2_start_index:L2_end_index]
        L2_Pf_window = L2_Pf[L2_start_index:L2_end_index]


        HF_TimeTicks_window_shifted = HF_TimeTicks_window - HF_TimeTicks_window[0]
        L1_TimeTicks_window_shifted = L1_TimeTicks_window - HF_TimeTicks_window[0]
        L2_TimeTicks_window_shifted = L2_TimeTicks_window - HF_TimeTicks_window[0]
        tagged_event_middle_time = (interval[0] + interval[1])/2.0 - HF_TimeTicks_window[0]


        time_ticks = L2_TimeTicks_window_shifted
        vals = L2_Real_window

















        #         background_levels_array = [60, 75, 100, 150, 200, 300, 400]
        #         for bl in background_levels_array:
        #             try:
        #                 L1_Real_event_intervals, L1_Real_event_index_flag_array = event_detector(L1_TimeTicks_window_shifted, L1_Real_window, background_level=bl)
        #                 if L1_Real_event_intervals.size != 0:
        #                     break
        #             except:
        #                 continue
        #         
        #         for bl in background_levels_array:
        #             try:
        #                 L2_Real_event_intervals, L2_Real_event_index_flag_array = event_detector(L2_TimeTicks_window_shifted, L2_Real_window, background_level=bl)
        #                 if L2_Real_event_intervals.size != 0:
        #                     break
        #             except:
        #                 continue
        L1_Real_event_intervals = event_detector2(L1_TimeTicks_window_shifted, L1_Real_window)
        L2_Real_event_intervals = event_detector2(L2_TimeTicks_window_shifted, L2_Real_window)

        if len(L1_Real_event_intervals) == 0 and len(L2_Real_event_intervals) == 0:
            event_intervals = array([[interval[0] - HF_TimeTicks_window[0],  interval[1] - HF_TimeTicks_window[0]]])
        else:        
            event_intervals = []
            for event_interval in L1_Real_event_intervals:
                event_intervals.append(event_interval)
            for event_interval in L2_Real_event_intervals:
                event_intervals.append(event_interval)
            event_intervals = np.array(event_intervals)

        if len(event_intervals.shape)>1 and len(event_intervals)>1:
            detected_events_with_temporal_metric = []
            for event_interval in event_intervals:
                time_offset = abs(event_interval.sum()/2 - tagged_event_middle_time)
                time_width = event_interval[1]-event_interval[0]
                metric = abs(event_interval.sum()/2 - tagged_event_middle_time) / time_width
                # Metric is abs time distance from middle of tagged time, divided by the width
                # We want to pick the event which is closest to the center, but also the widest
                detected_events_with_temporal_metric.append([metric, event_interval])
            detected_events_with_temporal_metric.sort()
            detected_event_interval = detected_events_with_temporal_metric[0][1]
        else:
            detected_event_interval = event_intervals




        L1_L2_plot_data = ((L1_TimeTicks_window_shifted, L1_Real_window),
                           (L1_TimeTicks_window_shifted, L1_Imag_window),
                           (L1_TimeTicks_window_shifted, L1_Amp_window),
                           (L2_TimeTicks_window_shifted, L2_Real_window),
                           (L2_TimeTicks_window_shifted, L2_Imag_window),
                           (L2_TimeTicks_window_shifted, L2_Amp_window),
                           (L1_TimeTicks_window_shifted, L1_Pf_window[:,0]),
                           (L1_TimeTicks_window_shifted, L1_Pf_window[:,1]),
                           (L1_TimeTicks_window_shifted, L1_Pf_window[:,2]),
                           (L1_TimeTicks_window_shifted, L1_Pf_window[:,3]),
                           (L1_TimeTicks_window_shifted, L1_Pf_window[:,4]),
                           (L1_TimeTicks_window_shifted, L1_Pf_window[:,5]),
                           (L2_TimeTicks_window_shifted, L2_Pf_window[:,0]),
                           (L2_TimeTicks_window_shifted, L2_Pf_window[:,1]),
                           (L2_TimeTicks_window_shifted, L2_Pf_window[:,2]),
                           (L2_TimeTicks_window_shifted, L2_Pf_window[:,3]),
                           (L2_TimeTicks_window_shifted, L2_Pf_window[:,4]),
                           (L2_TimeTicks_window_shifted, L2_Pf_window[:,5]))
           
        L1_L2_plot_labels = ("L1_Real", "L1_Imag", "L1_Amp", "L2_Real", "L2_Imag", "L2_Amp",
            "L1_Pf0", "L1_Pf1", "L1_Pf2", "L1_Pf3", "L1_Pf4", "L1_Pf5",
            "L2_Pf0", "L2_Pf1", "L2_Pf2", "L2_Pf3", "L2_Pf4", "L2_Pf5")



        before_event_time = detected_event_interval[0] - 5.0 # 5 second leeway
        after_event_time = detected_event_interval[1] + 5.0 # 5 second leeway
        middle_event_time = detected_event_interval.sum()/2


        before_event_HF_Time_index = where(abs(HF_TimeTicks_window_shifted - before_event_time) == min(abs(HF_TimeTicks_window_shifted - before_event_time)))[0] + HF_start_index
        middle_event_HF_Time_index = where(abs(HF_TimeTicks_window_shifted - middle_event_time) == min(abs(HF_TimeTicks_window_shifted - middle_event_time)))[0] + HF_start_index
        after_event_HF_Time_index = where(abs(HF_TimeTicks_window_shifted - after_event_time) == min(abs(HF_TimeTicks_window_shifted - after_event_time)))[0] + HF_start_index

        # Create the big plot

        fig = plt.figure(figsize=(20,18))
        ax0 = plt.subplot2grid((10,4), (0,0), rowspan=6)

        ax1 = plt.subplot2grid((10,4), (0, 1))
        ax2 = plt.subplot2grid((10,4), (1, 1))
        ax3 = plt.subplot2grid((10,4), (2, 1))
        ax4 = plt.subplot2grid((10,4), (3, 1))
        ax5 = plt.subplot2grid((10,4), (4, 1))
        ax6 = plt.subplot2grid((10,4), (5, 1))

        ax7 = plt.subplot2grid((10,4),  (0, 2))
        ax8 = plt.subplot2grid((10,4),  (1, 2))
        ax9 = plt.subplot2grid((10,4),  (2, 2))
        ax10 = plt.subplot2grid((10,4), (3, 2))
        ax11 = plt.subplot2grid((10,4), (4, 2))
        ax12 = plt.subplot2grid((10,4), (5, 2))

        ax13 = plt.subplot2grid((10,4), (0, 3))
        ax14 = plt.subplot2grid((10,4), (1, 3))
        ax15 = plt.subplot2grid((10,4), (2, 3))
        ax16 = plt.subplot2grid((10,4), (3, 3))
        ax17 = plt.subplot2grid((10,4), (4, 3))
        ax18 = plt.subplot2grid((10,4), (5, 3))

        ax19 = plt.subplot2grid((10,4), (6, 0), rowspan=2, colspan=4)
        ax20 = plt.subplot2grid((10,4), (8, 0), rowspan=2, colspan=4)

        axes=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14,
              ax15, ax16, ax17, ax18]


        plot_title = (r"Investigating Belkin Energy Data for " + house_dir[-3:-1] + " data file '" + tagged_training_filename + "'\nAppliance '" + appliance_name + "' from %.2f to %.2f seconds" % (start_time, end_time))
        fig.suptitle(plot_title, fontsize=22)

        ax0.imshow(HF[:,HF_start_index:HF_end_index], origin="lower", interpolation="nearest",
            extent=[HF_TimeTicks_window_shifted[0], HF_TimeTicks_window_shifted[-1], 0, 4096], 
            aspect=3.0*num_HF_timestamps/4096.0)
    
        ax0.plot([before_event_time, before_event_time], [0, 4096], color="red")
        ax0.plot([middle_event_time, middle_event_time], [0, 4096], color="green")
        ax0.plot([after_event_time, after_event_time], [0, 4096], color="blue")

        ax0.set_xlim(HF_TimeTicks_window_shifted[0], HF_TimeTicks_window_shifted[-1])
        ax0.set_ylim(0, 4096)
        ax0.set_xlabel("Time From Start [s]")
        ax0.set_ylabel("FFT Vector (Frequency-space)")
        ax0.set_title("Spectrogram of High Frequency Noise")


        for n in range(len(axes)):
            axes[n].plot(L1_L2_plot_data[n][0], L1_L2_plot_data[n][1], c="k", label=L1_L2_plot_labels[n])
            axes[n].plot([before_event_time, before_event_time], [L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1], color="red")
            axes[n].plot([middle_event_time, middle_event_time], [L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1], color="green")
            axes[n].plot([after_event_time, after_event_time], [L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1], color="blue")
            # axes[n].legend(loc="upper left")
            axes[n].set_ylabel(L1_L2_plot_labels[n])
            axes[n].set_xlim(HF_TimeTicks_window_shifted[0], HF_TimeTicks_window_shifted[-1])
            axes[n].set_ylim(L1_L2_plot_data[n][1].min()*0.9, L1_L2_plot_data[n][1].max()*1.1)
            if n in [5, 11, 17]:
                axes[n].set_xlabel("Time From Start [s]")
            else:
                setp(axes[n].get_xticklabels(), visible=False)

        before_average_spectrum = HF[:,before_event_HF_Time_index-6:before_event_HF_Time_index].sum(axis=1) / float(HF[:,before_event_HF_Time_index-6:before_event_HF_Time_index].shape[1])

        event_average_spectrum = HF[:,middle_event_HF_Time_index-3:middle_event_HF_Time_index+3].sum(axis=1) / float(HF[:,middle_event_HF_Time_index-3:middle_event_HF_Time_index+3].shape[1])

        after_average_spectrum = HF[:,after_event_HF_Time_index-6:after_event_HF_Time_index].sum(axis=1) / float(HF[:,after_event_HF_Time_index-6:after_event_HF_Time_index].shape[1])

        diff_before_spectrum = event_average_spectrum - before_average_spectrum
        diff_after_spectrum = event_average_spectrum - after_average_spectrum

        #         smooth_off_average_spectrum = ndimage.filters.gaussian_filter1d(off_average_spectrum, 5)
        #         smooth_on_average_spectrum = ndimage.filters.gaussian_filter1d(on_average_spectrum, 5)
        #         smooth_diff_spectrum = smooth_on_average_spectrum - smooth_off_average_spectrum

        ax19.plot(before_average_spectrum, color="red", label="Before")
        ax19.plot(event_average_spectrum, color="green", label="During")
        ax19.plot(after_average_spectrum, color="blue", label="After")

        ax19.set_xlim(0,4096)
        ax19.set_ylim(0,255)
        setp(ax19.get_xticklabels(), visible=False)
        ax19.legend(loc="lower left")
        ax20.plot(diff_before_spectrum, color="brown", label="D-B")
        ax20.plot(diff_after_spectrum, color="darkcyan", label="D-A")

        ax20.legend(loc="upper left")
        ax20.set_xlim(0,4096)
        ax20.set_ylabel("Difference")
        ax20.set_xlabel("FFT Vector (Frequency-space)")

        fig.subplots_adjust(hspace=0.175, wspace=0.25)

        canvas = FigureCanvas(fig)
        canvas.print_figure("plots/" + house_dir[-3:-1] + "_" + str(int(round((start_time + end_time)/2.0))) + "_" + appliance_name.replace(" ", "") + ".png", dpi=300, bbox_inches='tight')
        close("all")
