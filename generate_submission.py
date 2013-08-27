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

    #Do something with data
    with open('H4_apID_2_Testing_09_13_1347519601.txt','r') as first:
        reader = csv.reader(first)
        list1 = [int(round(float(time[0]))) for time in reader]
    with open('H4_apID_2_Testing_09_18_1347951601.txt','r') as second:
        reader = csv.reader(second)
        list2 = [int(round(float(time[0]))) for time in reader]
    print list1[0],list2[0]
    startLine = -1
    endLine = -1
    for ii,line in enumerate(subItems):
        if startLine == -1 and line[1] == 'H4' and line[2] == '2':
            startLine = ii
        elif line[1] == 'H4' and line[2] == '2':
            endLine = ii+1
    print startLine,endLine
    for ii in xrange(startLine,endLine):
        for jj in xrange(len(list1)):
            if list1[jj] >= int(round(float(subItems[ii][3]))) and list1[jj] < int(round(float(subItems[ii+1][3]))):
                subItems[ii][4] = '1'
            if list2[jj] >= int(round(float(subItems[ii][3]))) and list2[jj] < int(round(float(subItems[ii+1][3]))):
                subItems[ii][4] = '1'
    if note == '':
        base = 'submission'
    else:
        base = 'submission_'
    with open(base+note+'.csv','w+b') as submission:
        writer = csv.writer(submission,delimiter=',')
        for item in subItems:
            writer.writerow(item)

if __name__ == "__main__":
    main(sys.argv[1:])
