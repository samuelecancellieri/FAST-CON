#!/usr/bin/env python

import sys
import os

fileSet = set()
resultsDict = {}

destination = sys.argv[1]
saveName = str(destination)
saveName = saveName.strip().split('/')

for files in os.listdir(destination):
    if str(sys.argv[2]) in files:
        fields = files.strip().split('.')
        fileSet.add(str(fields[0]))
    else:
        continue

fileSet = sorted(fileSet)

for elem in fileSet:
    resultsDict[str(elem)] = {}
    resultsDict[str(elem)]['128'] = {}
    resultsDict[str(elem)]['256'] = {}
    resultsDict[str(elem)]['512'] = {}
    resultsDict[str(elem)]['1024'] = {}

for timeFile in os.listdir(destination):
    inTimeFile = open(destination + timeFile, "r")
    name = timeFile.rstrip().split('.')
    for line in inTimeFile:
        if 'mean_time' in line:
            fields = line.rstrip().split()
            totalTime = fields[1]
            resultsDict[str(name[0])][str(name[4])] = totalTime #cambiare name[>0] per definire la posizione del numero di thread nel file e.g. graph.txt.256 OR graph.txt.mtx.gg.256

resultFile = open('128.'+saveName[1]+'.'+str(sys.argv[2])+'.results.tsv','w')
for elem in fileSet:
    resultFile.write(str(elem)+'\t')
    if resultsDict[str(elem)]['128']:
        resultFile.write(str(round(float(resultsDict[str(elem)]['128'])*float(1000),2)))
    resultFile.write('\n')

resultFile = open('256.'+saveName[1]+'.'+str(sys.argv[2])+'.results.tsv','w')
for elem in fileSet:
    resultFile.write(str(elem)+'\t')
    if resultsDict[str(elem)]['256']:
        resultFile.write(str(round(float(resultsDict[str(elem)]['256'])*float(1000),2)))
    resultFile.write('\n')

resultFile = open('512.'+saveName[1]+'.'+str(sys.argv[2])+'.results.tsv','w')
for elem in fileSet:
    resultFile.write(str(elem)+'\t')
    if resultsDict[str(elem)]['512']:
        resultFile.write(str(round(float(resultsDict[str(elem)]['512'])*float(1000),2)))
    resultFile.write('\n')

resultFile = open('1024.'+saveName[1]+'.'+str(sys.argv[2])+'.results.tsv','w')
for elem in fileSet:
    resultFile.write(str(elem)+'\t')
    if resultsDict[str(elem)]['1024']:
        resultFile.write(str(round(float(resultsDict[str(elem)]['1024'])*float(1000),2)))
    resultFile.write('\n')
