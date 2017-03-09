# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:17:58 2017

@author: Mark Schulze
"""

import os
import time
import numpy as np
import scipy.misc
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt
#from scipy.cluster.vq import kmeans, whiten

from sklearn import preprocessing

# base directory for feature vectors, organized in subdirectories by name
BASE_DIR = '/Users/Mark/Desktop/Work/Waldo Photos/lfw_data/lfw_vectors/'

# specify which starting letter(s) of names to use, if empty use all data
STARTS_WITH = ['A', 'J', 'M']
#STARTS_WITH = ['A']
#STARTS_WITH = []

SHOW_PLOT = True
SHOW_PRC = True
thresholdList = np.arange(1.15, 1.31, 0.02).tolist()
thresholdList = [0.6, 0.8, 0.9] + thresholdList + [1.4, 1.5, 1.6]
#thresholdList = [1.10]

# set up plots
if SHOW_PLOT:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Histogram of cluster sizes (log-log)')
    ax1.set_xlabel('Bin center (number of points in cluster)')
    ax1.set_ylabel('Number of clusters of this size')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('Histogram of cluster size (semilog)')
    ax2.set_xlabel('Number of points in cluster (bin)')
    ax2.set_ylabel('Percentage of clusters that are this size')
    
    if SHOW_PRC:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3b = ax3.twinx()
        ax3.set_title('Precision-Recall Curve')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision', color='b')
        ax3b.set_ylabel('F-score', color='g')

def clusterSizes(clLabels):
    # returns list of cluster sizes from a label list
    sizeList = []
#    print np.max(clLabels)
    for idNum in range(0, np.max(clLabels) + 1):
        sizeList.append(clLabels.count(idNum))
    return sizeList
    

def maxLabels(clLabels):
    # returns the number of occurrences of the label in a list that occurs most often
    labelCount = {}
    for label in clLabels:
        try:
            labelCount[label] += 1
        except KeyError:
            labelCount[label] = 1
    return np.max([x for x in labelCount.values()])

def sameIDPairs(clLabels):
    # returns total number of matching pairs in a set of labels
    # singletons count as one matching pair
    numSameIDPairs = 0
    for idNum in range(0, np.max(clLabels) + 1):
        numInCl = clLabels.count(idNum)
        if numInCl > 1:
#            print idNum, ":", numInCl
            numSameIDPairs += scipy.misc.comb(numInCl, 2)
        elif numInCl == 1:
            numSameIDPairs += 1
    return numSameIDPairs

def matchingPairs(inSet, otherSet):
    # returns number of pairs in each label of inSet that have matching values in the corresponding otherSet
    # singletons count as one matching pair
    matchingPairs = 0
    for inSetNum in range(1, np.max(inSet) + 1):
        inSetLocs = np.where(inSet == inSetNum)[0]
#        print inSetLocs.tolist()
        otherLabelsOfInSet = [otherSet[x] for x in inSetLocs]
        if len(otherLabelsOfInSet) > 1:
            matchingPairs += scipy.misc.comb(maxLabels(otherLabelsOfInSet), 2)
        elif len(otherLabelsOfInSet) == 1:
            matchingPairs += 1
    return matchingPairs


# read vectors from files and store in vec
vec = []
trueClasses = []
classNum = 0
for base, subs, files in os.walk(BASE_DIR):
    for filename in files:
        if filename.endswith('.npy') and (len(STARTS_WITH) < 1 or filename[0] in STARTS_WITH):
            vec.append(np.load(os.path.join(base, filename)))
            trueClasses.append(classNum)
    classNum += 1       # when we switch to a new folder, that's a new class

# normalize the data
data = preprocessing.normalize(vec)

nSamples, nFeatures = data.shape
print nSamples, nFeatures

trueClusterSizes = clusterSizes(trueClasses)
histBins = [0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5]    # histogram bin edges
histCenters = [1.0, 2.0, 3.5, 6.5, 12.5, 24.5]      # histogram bin centers

# set up plots
if SHOW_PLOT:
    ax2.hist(trueClusterSizes, histBins, normed=1, alpha=0.5, label='truth')
    fig2.gca().set_yscale("log")
    trueHist = np.histogram(trueClusterSizes, bins=histBins)
    trueHistogram = np.double(trueHist[0]) #/ np.sum(trueHist[0])
    print "True cluster histogram:", trueHistogram
    ax1.loglog(histCenters, trueHistogram, basex=2, label='truth')

precisionList = []
recallList = []
FList = []
for threshold in thresholdList:
    start_time = time.time()
    print "Starting clustering..."
    clusters = hcluster.fclusterdata(data, threshold, criterion='distance', metric='euclidean', method='average')
    print("Done clustering in %s seconds" % (time.time() - start_time))
    
    # K-means clustering attempt
#    centroids = kmeans(whitened, 300)[0]
#    clusters = []
#    for datum in whitened:
#        minDist = np.double.max
#        for cidx in range(0, len(centroids)):
#            dist = np.linalg.norm(datum - centroids[cidx])
#            if dist < minDist:
#                minDist = dist
#                clNum = cidx
#        clusters.append(clNum)
#    clusters = np.asarray(clusters)

    # get cluster sizes and plot histograms    
    calcClusterSizes = clusterSizes(clusters.tolist())
    if SHOW_PLOT:
        ax2.hist(calcClusterSizes, histBins, normed=1, alpha=0.25, label='calculated')
        calcHist = np.histogram(calcClusterSizes, bins=histBins)
        calcHistogram = np.double(calcHist[0]) #/ np.sum(calcHist[0])
        ax1.loglog(histCenters, calcHistogram, basex=2, label='calculated')
        ax1.legend()
        ax2.legend()
    print "Threshold ", threshold, ":", calcHistogram

    # calculate precision
    matchingSameClusterPairs = matchingPairs(clusters, trueClasses)
    print "matching same cluster pairs =", matchingSameClusterPairs
    totalSameClusterPairs = sameIDPairs(clusters.tolist())
    print "total same cluster pairs =", totalSameClusterPairs
    precision = matchingSameClusterPairs / totalSameClusterPairs
    print "Precision =", precision
    
    # calculate recall
    matchingSameClassPairs = matchingPairs(np.asarray(trueClasses), clusters)
    print "matching same class pairs =", matchingSameClassPairs
    totalSameClassPairs = sameIDPairs(trueClasses)
    print "total same class pairs =", totalSameClassPairs
    recall = matchingSameClassPairs / totalSameClassPairs
    print "Recall =", recall
    
    # calculate F-score
    F = 2 * precision * recall / (precision + recall)
    print "F =", F
    
    precisionList.append(precision)
    recallList.append(recall)
    FList.append(F)

if SHOW_PLOT:
    if SHOW_PRC:
        ax3.plot(recallList, precisionList, color='b', linewidth=2)
        ax3b.plot(recallList, FList, color='g', linewidth=2)
    plt.show()