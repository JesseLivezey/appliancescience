import ApplianceFeatures as af
import numpy as np
from scipy import ndimage, signal
from datetime import timedelta

"""
This just reads in the appliance training data and measures the jump in a given
data stream (Real Power, Imaginary Power, Amplitude, or Power Factor). The
"jump" information calculated is the difference in the stream when the appliance
is "on" vs when it is "off" (the baseline). Additionally, the standard deviation
of the jump is reported. This works well for simple appliances with only on and
off states. It will not be useful for appliances that change their load during
operation.
"""




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

def extract_event_value(stream):
# L1 and L2 data is collected at a rate of 6.0064028254118895 times per second
# HF spectra data is collected at a rate of 0.9375005859378663 times per second
    median_stream = ndimage.filters.median_filter(stream, 6.0064028254118895) # smooth with width 1 second of data
    smooth_stream = ndimage.filters.gaussian_filter1d(median_stream, 1.0) # smooth
    stream_gradient = np.gradient(smooth_stream)
    peak_locs = signal.find_peaks_cwt(abs(stream_gradient), np.array([1]), min_snr=1)
    clipped_stream_gradient, stream_gradient_median, std_around_zero = mad_clipping(stream_gradient, 3)
    vetted_peak_locs = list(np.where(abs(stream_gradient[peak_locs]) > 5*std_around_zero)[0])
    if len(vetted_peak_locs) > 1:
        peak_indices = list(np.array(peak_locs)[vetted_peak_locs])
        peak_times = stream.index[peak_indices]
        peak_data = stream_gradient[peak_indices]
        precise_event_start_timestamp = peak_times.min()
        precise_event_end_timestamp = peak_times.max()
        buffer_time = timedelta(seconds=0.5)
        before_stream = stream[stream.index<(precise_event_start_timestamp-buffer_time)]
        during_stream = stream[(stream.index>(precise_event_start_timestamp+buffer_time)) & (stream.index<(precise_event_end_timestamp-buffer_time))]
        after_stream = stream[stream.index>(precise_event_end_timestamp+buffer_time)]
        baseline_std = ((before_stream.std())**2 + (after_stream.std())**2)**0.5
        if not (abs(before_stream.median() - after_stream.median()) < 2*baseline_std):
            # print "OMG, before_stream and after_stream indicate a baseline that changes during the event. Abort!!!"
            return None, None
        else:
            baseline_value = (before_stream.median() + after_stream.median())/2.0
            cropped_during_stream, event_value, event_std = mad_clipping(during_stream, 3)
            event_difference = event_value - baseline_value
            event_different_std = ((baseline_std)**2 + (event_std)**2)**0.5
            return event_difference, event_different_std
    else:
        # print "OMG, there were not at least 2 detected jumps in the stream, so no event can be measured."
        return None, None



# appliance_event_number = 0 # there are 441

data_list = []
for appliance_event_number in range(100, 115):
    app=af.Appliance(appliance_event_number)

    L1_Real_val, L1_Real_std = extract_event_value(app.l1_event.real.sum(axis=1))
    L1_Imag_val, L1_Imag_std = extract_event_value(app.l1_event.imag.sum(axis=1))

    L2_Real_val, L2_Real_std = extract_event_value(app.l2_event.real.sum(axis=1))
    L2_Imag_val, L2_Imag_std = extract_event_value(app.l2_event.imag.sum(axis=1))

    data_list.append([app.id, app.house, app.name, L1_Real_val, L1_Real_std, L1_Imag_val, L1_Imag_std, L2_Real_val, L2_Real_std, L2_Imag_val, L2_Imag_std])

for line in data_list:
    print line[0], "\t", line[1], line[2], "[(", line[3], "+/-", line[4], ") (", line[5], "+/-", line[6], ")]", "[(", line[7], "+/-", line[8], ") (", line[9], "+/-", line[10], ")]"


# print "L1 Real 0:", extract_event_value(app.l1_event.real["0"])
# print "L1 Real 1:", extract_event_value(app.l1_event.real["1"])
# print "L1 Real 2:", extract_event_value(app.l1_event.real["2"])
# print "L1 Real 3:", extract_event_value(app.l1_event.real["3"])
# print "L1 Real 4:", extract_event_value(app.l1_event.real["4"])
# print "L1 Real 5:", extract_event_value(app.l1_event.real["5"])
# 
# print "L1 Imag 0:", extract_event_value(app.l1_event.imag["0"])
# print "L1 Imag 1:", extract_event_value(app.l1_event.imag["1"])
# print "L1 Imag 2:", extract_event_value(app.l1_event.imag["2"])
# print "L1 Imag 3:", extract_event_value(app.l1_event.imag["3"])
# print "L1 Imag 4:", extract_event_value(app.l1_event.imag["4"])
# print "L1 Imag 5:", extract_event_value(app.l1_event.imag["5"])
# 
# print "L1 Amp 0:", extract_event_value(app.l1_event.amp["0"])
# print "L1 Amp 1:", extract_event_value(app.l1_event.amp["1"])
# print "L1 Amp 2:", extract_event_value(app.l1_event.amp["2"])
# print "L1 Amp 3:", extract_event_value(app.l1_event.amp["3"])
# print "L1 Amp 4:", extract_event_value(app.l1_event.amp["4"])
# print "L1 Amp 5:", extract_event_value(app.l1_event.amp["5"])
# 
# print "L1 PF 0:", extract_event_value(app.l1_event.pf["0"])
# print "L1 PF 1:", extract_event_value(app.l1_event.pf["1"])
# print "L1 PF 2:", extract_event_value(app.l1_event.pf["2"])
# print "L1 PF 3:", extract_event_value(app.l1_event.pf["3"])
# print "L1 PF 4:", extract_event_value(app.l1_event.pf["4"])
# print "L1 PF 5:", extract_event_value(app.l1_event.pf["5"])
# 
# print "L2 Real 0:", extract_event_value(app.l2_event.real["0"])
# print "L2 Real 1:", extract_event_value(app.l2_event.real["1"])
# print "L2 Real 2:", extract_event_value(app.l2_event.real["2"])
# print "L2 Real 3:", extract_event_value(app.l2_event.real["3"])
# print "L2 Real 4:", extract_event_value(app.l2_event.real["4"])
# print "L2 Real 5:", extract_event_value(app.l2_event.real["5"])
# 
# print "L2 Imag 0:", extract_event_value(app.l2_event.imag["0"])
# print "L2 Imag 1:", extract_event_value(app.l2_event.imag["1"])
# print "L2 Imag 2:", extract_event_value(app.l2_event.imag["2"])
# print "L2 Imag 3:", extract_event_value(app.l2_event.imag["3"])
# print "L2 Imag 4:", extract_event_value(app.l2_event.imag["4"])
# print "L2 Imag 5:", extract_event_value(app.l2_event.imag["5"])
# 
# print "L2 Amp 0:", extract_event_value(app.l2_event.amp["0"])
# print "L2 Amp 1:", extract_event_value(app.l2_event.amp["1"])
# print "L2 Amp 2:", extract_event_value(app.l2_event.amp["2"])
# print "L2 Amp 3:", extract_event_value(app.l2_event.amp["3"])
# print "L2 Amp 4:", extract_event_value(app.l2_event.amp["4"])
# print "L2 Amp 5:", extract_event_value(app.l2_event.amp["5"])
# 
# print "L2 PF 0:", extract_event_value(app.l2_event.pf["0"])
# print "L2 PF 1:", extract_event_value(app.l2_event.pf["1"])
# print "L2 PF 2:", extract_event_value(app.l2_event.pf["2"])
# print "L2 PF 3:", extract_event_value(app.l2_event.pf["3"])
# print "L2 PF 4:", extract_event_value(app.l2_event.pf["4"])
# print "L2 PF 5:", extract_event_value(app.l2_event.pf["5"])