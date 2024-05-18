###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Feb 12th 2021                                                 #
###########################################################################################

####################################################################################################
#                                                                                                  #
# THE CURRENT VERSION WAS UPDATED WITH A VISUAL INTERFACE, INCLUDING MORE METRICS AND SUPPORTING   #
# OTHER FILE FORMATS. PLEASE ACCESS IT ACCESSED AT:                                                #
#                                                                                                  #
# https://github.com/rafaelpadilla/review_object_detection_metrics                                 #
#                                                                                                  #
# @Article{electronics10030279,                                                                    #
#     author         = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto,    #
#                       Sergio L. and da Silva, Eduardo A. B.},                                    #
#     title          = {A Comparative Analysis of Object Detection Metrics with a Companion        #
#                       Open-Source Toolkit},                                                      #
#     journal        = {Electronics},                                                              #
#     volume         = {10},                                                                       #
#     year           = {2021},                                                                     #
#     number         = {3},                                                                        #
#     article-number = {279},                                                                      #
#     url            = {https://www.mdpi.com/2079-9292/10/3/279},                                  #
#     issn           = {2079-9292},                                                                #
#     doi            = {10.3390/electronics10030279}, }                                            #
####################################################################################################

####################################################################################################
# If you use this project, please consider citing:                                                 #
#                                                                                                  #
# @INPROCEEDINGS {padillaCITE2020,                                                                 #
#    author    = {R. {Padilla} and S. L. {Netto} and E. A. B. {da Silva}},                         #
#    title     = {A Survey on Performance Metrics for Object-Detection Algorithms},                #
#    booktitle = {2020 International Conference on Systems, Signals and Image Processing (IWSSIP)},#
#    year      = {2020},                                                                           #
#    pages     = {237-242},}                                                                       #
#                                                                                                  #
# This work is published at: https://github.com/rafaelpadilla/Object-Detection-Metrics             #
####################################################################################################

import argparse
import glob
import os
import shutil
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils_pascal import BBFormat

import ipdb
st = ipdb.set_trace

currentPath = '/home/gsarch/repo/project_cleanup/Object-Detection-Metrics'


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append('argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' %
                      argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append('%s. It must be in the format \'width,height\' (e.g. \'600,400\')' %
                          errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg

def add_bounding_box(bbox_params, allBoundingBoxes, allClasses, nameOfImage, isGT, imgSize, Format, CoordType):
    if isGT:
        # idClass = int(splitLine[0]) #class
        idClass = (bbox_params[0])  # class
        x = float(bbox_params[1])
        y = float(bbox_params[2])
        w = float(bbox_params[3])
        h = float(bbox_params[4])
        bb = BoundingBox(nameOfImage,
                            idClass,
                            x,
                            y,
                            w,
                            h,
                            CoordType,
                            imgSize,
                            BBType.GroundTruth,
                            format=Format)
    else:
        # idClass = int(splitLine[0]) #class
        idClass = (bbox_params[0])  # class
        confidence = float(bbox_params[1])
        x = float(bbox_params[2])
        y = float(bbox_params[3])
        w = float(bbox_params[4])
        h = float(bbox_params[5])
        bb = BoundingBox(nameOfImage,
                            idClass,
                            x,
                            y,
                            w,
                            h,
                            CoordType,
                            imgSize,
                            BBType.Detected,
                            confidence,
                            format=Format)
    allBoundingBoxes.addBoundingBox(bb)
    if idClass not in allClasses:
        allClasses.append(idClass)
    return allBoundingBoxes, allClasses

def get_map(allBoundingBoxes, allClasses, IOU_threshold=0.5):
    # Get current path to set default folders
    # currentPath = os.path.dirname(os.path.abspath(__file__))

    VERSION = '0.2 (beta)'

    iouThreshold = IOU_threshold
    
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0
    acc_AR = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=None,
        showGraphic=False)

    # f = open(os.path.join(savePath, 'results.txt'), 'w')
    # f.write('Object Detection Metrics\n')
    # f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    # f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    classes = []
    # precs = []
    # recs = []
    aps = []
    ars = []
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        ar = np.nanmean(np.array(recall))
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            if not np.isnan(ar):
                acc_AR = acc_AR + ar
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            # print('AP: %s (%s)' % (ap_str, cl))
            # f.write('\n\nClass: %s' % cl)
            # f.write('\nAP: %s' % ap_str)
            # f.write('\nPrecision: %s' % prec)
            # f.write('\nRecall: %s' % rec)
            classes.append(cl)
            ars.append(ar * 100)
            aps.append(ap * 100)
    # if validClasses==0:
    #     validClasses = 1 # otherwise error
    mAP = acc_AP / validClasses
    mAR = acc_AR / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    # print('mAP: %s' % mAP_str)
    # f.write('\n\n\nmAP: %s' % mAP_str)

    return mAP, classes, aps, ars, mAR
