import ApplianceFeatures as ap

import numpy as np

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime, timedelta
import os, sys


transition_filename = "all_transitions.csv"
# transition_filename = "simple_features_H1_07-11_testing.csv"
transition_data = np.loadtxt(transition_filename, skiprows=1, delimiter=",",
    dtype=[ ("timestamp",np.float64),
            ("L1_real",np.float32), 
            ("L1_imag",np.float32), 
            ("L2_real",np.float32),
            ("L2_imag",np.float32) ] )


plot_height = 5000

file_list = os.listdir("data/hdf5storage/")
# file_list = ["H1_07-11_testing.h5"]
for datafilename in file_list:
    if datafilename[-3:] != ".h5":
        continue
        
    a = ap.ElectricTimeStream("data/hdf5storage/" + datafilename)
    # Extract Components on the ap object
    a.ExtractComponents()

    # Trim to specified time range. Can take date-time object or strings
    # a_trimmed = a.l1.amp["0"][(a.l1.index>"2012-04-13 22:04:51.253961") &(a.l1.index<"2012-04-13 22:05:51.253961")]

    L1_Amp = a.l1.amp.sum(axis=1)
    L2_Amp = a.l2.amp.sum(axis=1)

    L1_time_length = (L1_Amp.index[-1] - L1_Amp.index[0]).total_seconds()
    L2_time_length = (L2_Amp.index[-1] - L2_Amp.index[0]).total_seconds()

    n_plot_rows_L1 = ceil(L1_time_length/1800.)
    n_plot_rows_L2 = ceil(L2_time_length/1800.)
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
        
        
        
        
        plot_start_time = min(L2_time_start, L1_time_start)
        plot_end_time = max(L2_time_end, L1_time_end)
        
        plot_start_time_seconds = (plot_start_time - datetime(1970,1,1)).total_seconds()
        plot_end_time_seconds = (plot_end_time - datetime(1970,1,1)).total_seconds()
        
        plot_transitions = transition_data[ (transition_data["timestamp"]>plot_start_time_seconds) & (transition_data["timestamp"]<plot_end_time_seconds) ]
        for transition in plot_transitions:
            transition_timestamp = datetime(1970,1,1) + timedelta(seconds=transition["timestamp"])
            desc_text = ""
            if transition["L1_real"] > 15:
                desc_text += "L1R: +%.0f, " % transition["L1_real"]
            if transition["L1_real"] < -15:
                desc_text += "L1R: %.0f, " % transition["L1_real"]
            if transition["L1_imag"] > 15:
                desc_text += "L1I: +%.0f, " % transition["L1_imag"]
            if transition["L1_imag"] < -15:
                desc_text += "L1I: %.0f, " % transition["L1_imag"]
            if transition["L2_real"] > 15:
                desc_text += "L2R: +%.0f, " % transition["L2_real"]
            if transition["L2_real"] < -15:
                desc_text += "L2R: %.0f, " % transition["L2_real"]
            if transition["L2_imag"] > 15:
                desc_text += "L2I: +%.0f, " % transition["L2_imag"]
            if transition["L2_imag"] < -15:
                desc_text += "L2I: %.0f, " % transition["L2_imag"]
            desc_text = desc_text.rstrip(", ")
            transition_line_color = "red"
            if (transition["L1_real"]**2+transition["L1_imag"]**2)**0.5 > (transition["L2_real"]**2+transition["L2_imag"]**2)**0.5:
                desc_text_color = "purple"
                if transition["L1_real"] + transition["L1_imag"] > 0:
                    transition_line_color = "green"
            else:
                desc_text_color = "orange"
                if transition["L2_real"] + transition["L2_imag"] > 0:
                    transition_line_color = "green"
            
            axes[n].plot([transition_timestamp, transition_timestamp], [0, plot_height], color=transition_line_color, lw=2)
            if transition_line_color == "green":
                axes[n].text(transition_timestamp + timedelta(seconds=2), plot_height-100, desc_text, va="top", ha="left", color=desc_text_color, rotation=90)
                axes[n].scatter([transition_timestamp], [plot_height-50], marker="^", color="green", s=80)
            else:
                axes[n].text(transition_timestamp - timedelta(seconds=2), plot_height-100, desc_text, va="top", ha="right", color=desc_text_color, rotation=90)
                axes[n].scatter([transition_timestamp], [plot_height-50], marker="v", color="red", s=80)
                            
                
        
        axes[n].set_xlim(plot_start_time, plot_end_time)
        axes[n].set_ylim(0,plot_height)
        axes[n].set_ylabel("Line Amplitudes")
        axes[n].set_xlabel("Timestamp")
    
    canvas = FigureCanvas(fig)
    canvas.print_figure("full_plots/" + datafilename.split(".")[0] + ".png", dpi=72, bbox_inches='tight')
    close("all")