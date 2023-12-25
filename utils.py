import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.nn.functional as F

random.seed(42)

# function: parses date string into DateTime Object
# input: date
# output: DateTime Object
def dateParser(date):
    mainFormat = '%Y-%m-%d %H:%M:%S.%f'
    altFormat = '%Y-%m-%d %H:%M:%S'
    try:
        return datetime.datetime.strptime(date, mainFormat)
    except ValueError:
        return datetime.datetime.strptime(date, altFormat)

# function: return a DataFrame from directory
# input: file directory dexcom
# output: DataFrame
def getGlucoseData(fileDir):
    df = pd.read_csv(fileDir)
    data = pd.DataFrame()
    data['Time'] = df['Timestamp (YYYY-MM-DDThh:mm:ss)']
    data['Glucose'] = pd.to_numeric(df['Glucose Value (mg/dL)'])
    data.drop(data.index[:12], inplace=True)
    data['Time'] = np.array([dateParser(dateStr) for dateStr in data['Time']])
    data['Day'] = np.array([date.day for date in data['Time']])
    data = data.reset_index()
    return data

# function: create samples given glucose data
# input: data, length, numSamples
# output: np array of samples and value after
def sampleTransformer(data, length, numSamples=100):
    SOS_token = np.array([301])
    EOS_token = np.array([302])
    ans = []
    for i in range(numSamples):
        start = random.randint(0,len(data)- 2 * length - 1)
        while True in np.isnan(data[start: start + 2 * length + 1]):
            start = random.randint(0,len(data)-2 * length-1)
        begin = np.concatenate((SOS_token, data[start : start + length], EOS_token))
        end = np.concatenate((SOS_token, data[start + length : start + 2 * length], EOS_token))
        ans.append([begin, end])
    np.random.shuffle(ans)
    return np.array(ans)

# function: create samples for glucose personal values
# input: glucoseData, edaData, hrData, tempData, length, numSamples
# output: np array of samples and value after
def createSamples(glucoseData, edaData, hrData, tempData, pp5vals, length, numSamples=100):
    SOS_token = np.array([1001])
    EOS_token = np.array([1002])
    ans = []
    for i in range(numSamples):
        glucStart = random.randint(0,len(glucoseData) - length - 1)
        edaStart = glucStart * pp5vals.eda
        hrStart = glucStart * pp5vals.hr
        tempStart = glucStart * pp5vals.temp
        glucTruth = True in np.isnan(glucoseData[glucStart: glucStart + length])
        edaTruth = True in np.isnan(edaData[edaStart: edaStart + length * pp5vals.eda]) or len(edaData[edaStart: edaStart + length * pp5vals.eda]) != length * pp5vals.eda
        hrTruth = True in np.isnan(hrData[hrStart: hrStart + length * pp5vals.hr]) or len(hrData[hrStart: hrStart + length * pp5vals.hr]) != length * pp5vals.hr
        tempTruth = True in np.isnan(tempData[tempStart: tempStart + length * pp5vals.temp]) or len(tempData[tempStart: tempStart + length * pp5vals.temp]) != pp5vals.temp
        while glucTruth or edaTruth or hrTruth or tempTruth:
            glucStart = random.randint(0,len(glucoseData) - length - 1)
            edaStart = glucStart * pp5vals.eda
            hrStart = glucStart * pp5vals.hr
            tempStart = glucStart * pp5vals.temp
            glucTruth = True in np.isnan(glucoseData[glucStart: glucStart + length])
            edaTruth = True in np.isnan(edaData[edaStart: edaStart + length * pp5vals.eda]) or len(edaData[edaStart: edaStart + length * pp5vals.eda]) != length * pp5vals.eda
            hrTruth = True in np.isnan(hrData[hrStart: hrStart + length * pp5vals.hr]) or len(hrData[hrStart: hrStart + length * pp5vals.hr]) != length * pp5vals.hr
            tempTruth = True in np.isnan(tempData[tempStart: tempStart + length * pp5vals.temp]) or len(tempData[tempStart: tempStart + length * pp5vals.temp]) != length * pp5vals.temp
        eda = np.concatenate((SOS_token, avgSequence(edaData[edaStart : edaStart + length * pp5vals.eda], length, pp5vals.eda), EOS_token))
        hr = np.concatenate((SOS_token, avgSequence(hrData[hrStart : hrStart + length * pp5vals.hr], length, pp5vals.hr), EOS_token))
        temp = np.concatenate((SOS_token, avgSequence(tempData[tempStart : tempStart + length * pp5vals.temp], length, pp5vals.temp), EOS_token))
        # persMean = np.mean(glucoseData[glucStart : glucStart + length])
        # persStd = np.std(glucoseData[glucStart : glucStart + length])
        # persGluc = np.concatenate((SOS_token, [persComp(i, persMean, persStd) for i in glucoseData[glucStart : glucStart + length]], EOS_token))
        gluc = np.concatenate((SOS_token, glucoseData[glucStart: glucStart + length].astype(int), EOS_token))
        ans.append([np.stack([eda, hr, temp]), gluc])
    np.random.shuffle(ans)
    return ans

# function: returns 3 (persHigh), 2 (persNorm), or 1 (persLow)
def persComp(value, persMean, persStd):
    if value > persMean + persStd:
        return 3
    elif value < persMean - persStd:
        return 1
    return 2

def avgSequence(data, length, pp5):
    idx = pp5
    ret = []
    while idx <= length * pp5:
        ret.append(np.mean(data[idx - pp5: idx]))
        idx += pp5
    return np.array(ret)
    
    
# function: create a matrix of samples
# input: glucoseDict, length, numSamples
# output: train_data, val_data
def createSamplesArray(glucoseDict, length, numSamples):
    shuffled = list(glucoseDict.keys())
    np.random.shuffle(shuffled)
    train_choice, val_choice = (shuffled[:12], shuffled[12:])
    train_data = []
    val_data = []
    for i in train_choice:
        data = glucoseDict[i]
        train = sampleTransformer(data, length, numSamples)
        for trainVal in train:
            train_data.append(trainVal)
    for i in val_choice:
        data = glucoseDict[i]
        val = sampleTransformer(data, length, numSamples)
        for validVal in val:
            val_data.append(validVal)
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    return train_data, val_data

# function: create samples of glucose data regardless of which sample it is from
# input: glucoseDict, length, numSamples
# output: X, y
def createGlucoseSamples(glucoseDict, length, numSamples):
    data = sampleTransformer(glucoseDict[random.choice(list(glucoseDict.keys()))], length, numSamples)
    random.shuffle(data)
    return np.array([data[i][0] for i in range(len(data))]), np.array([data[i][1] for i in range(len(data))])

def createPersSamples(glucoseDict, edaDict, hrDict, tempDict, pp5vals, length, numSamples, samples):
    data = []
    for sample in samples:
        sampleNum = random.choice(list(glucoseDict.keys()))
        data.extend(createSamples(glucoseDict[sampleNum], edaDict[sampleNum], hrDict[sampleNum], tempDict[sampleNum], pp5vals, length, numSamples))
    random.shuffle(data)
    return np.array([np.array(data[i][0]).astype(np.int64) for i in range(len(data))]), np.array([data[i][1] for i in range(len(data))])