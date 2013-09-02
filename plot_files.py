import ApplianceFeatures as ap
from datetime import timedelta
import numpy as np

from matplotlib.pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os

# Read in Electric Time Stream

file_list = os.listdir("data/hdf5storage/")
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
        # axes[n].set_yscale("symlog")
        axes[n].set_ylim(10,4000)
        axes[n].set_ylabel("Line Amplitudes")
        axes[n].set_xlabel("Timestamp")
    
    canvas = FigureCanvas(fig)
    canvas.print_figure("full_plots/" + datafilename.split(".")[0] + ".png", dpi=72, bbox_inches='tight')
    close("all")