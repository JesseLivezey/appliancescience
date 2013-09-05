import numpy as np
import os



submission_data = np.loadtxt("data/SampleSubmission.csv", delimiter=",", skiprows=1, 
    dtype=[ ("line",np.int), ("house", "S2"), ("id", "S2"), ("timestamp", np.float64), ("pred", np.int)])


filelist = os.listdir("classified_series")
classified_interval_files = []
for filename in filelist:
    if "testing_classified_intervals.txt" in filename:
        classified_interval_files.append(filename)

for interval_file in classified_interval_files:
    interval_data = np.loadtxt("classified_series/" + interval_file, delimiter=",", 
        dtype=[ ("start",np.float64), ("stop", np.float64), ("house", "S2"), ("id", "S2")])
    if len(interval_data.shape) > 0:
        for interval in interval_data:
        
            # Rounding to activate a timestamp as if it were the full 60-s interval
            # start_rounded = round(interval["start"]) - np.mod(round(interval["start"]), 60) - 1 
            # stop_rounded = round(interval["stop"]) + (60-np.mod(round(interval["stop"]), 60)) + 1
            
            # Rounding to activate a timestamp only if that exact moment falls during the interval
            start_rounded = round(interval["start"])
            stop_rounded = round(interval["stop"])
            
            submission_data["pred"][np.where( (submission_data["timestamp"] > start_rounded) & (submission_data["timestamp"] < stop_rounded) &
                      (submission_data["house"] == interval["house"]) & (submission_data["id"] == interval["id"]))] = 1
    else:
        interval = interval_data
        start_rounded = round(interval["start"]) - np.mod(round(interval["start"]), 60) - 1 
        stop_rounded = round(interval["stop"]) + (60-np.mod(round(interval["stop"]), 60)) + 1
        submission_data["pred"][np.where( (submission_data["timestamp"] > start_rounded) & (submission_data["timestamp"] < stop_rounded) &
                  (submission_data["house"] == interval["house"]) & (submission_data["id"] == interval["id"]))] = 1


new_submission_file = file("AS_Submission.csv", "w")
new_submission_file.write("Id,House,Appliance,TimeStamp,Predicted\n")
for line in submission_data:
    new_submission_file.write(str(line["line"]) + "," + line["house"] + "," + line["id"] + "," + str(int(line["timestamp"])) + "," + str(line["pred"]) + "\n")
new_submission_file.close()