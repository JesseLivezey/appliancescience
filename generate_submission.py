import numpy as np
import csv
import argparse
import sys
import cPickle

def main(argv):
    #Parse file to generate submission with
    parser = argparse.ArgumentParser()
    parser.add_argument('eventFile')
    parser.add_argument('-n','--note',help='Note to append to filename')
    args = parser.parse_args(argv)
    eventFile = args.eventFile
    note = ''
    if args.note is not None:
        note = args.note
    print note
    print eventFile
    #Import sample submission to use as template
    with open(eventFile,'rb') as events:
        reader = csv.reader(events,delimiter=',')
        subItems = [item for item in reader]
    for ii in xrange(10):
        print subItems[ii]

    #Do something with data
    with open('H4_apID_2_Testing_09_13_1347519601.txt','rb') as first:
        reader = csv.reader(first)
        list1 = [int(round(float(time))) for time in reader]
    with open('H4_apID_2_Testing_09_18_1347951601.txt','rb') as second:
        reader = csv.reader(second)
        list2 = [int(round(float(time))) for time in reader]

    for ii,line in enumerate(subItems):
        pass

    if note == '':
        base = 'submission'
    else:
        base = 'submission_'
    with open(base+note+'.csv','wb') as submission:
        writer = csv.writer(submission,delimiter=',')
        for item in subItems:
            writer.writerow(item)

if __name__ == "__main__":
    main(sys.argv[1:])
