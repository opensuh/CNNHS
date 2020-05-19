import tensorflow as tf 
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math
from config import BATCH_SIZE, RATIO, BAND1, BAND2, BAND3
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import PCA
from random import sample
import pandas as pd
import csv
import cv2
import re
import os


def lowPass(signal, cut, sample_length, sample_rate):
  ''' 
  low pass filter
  args:
    signal = input signal
    cut = filtering bandwidth Hz
    sample_length = total signal length (number of samples)
    sample_rate = sampling rate of the input signal
  return:
    filtered.real = filtered signal
  '''
  ratio = int(sample_length/sample_rate)
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.signal.fft(signal)
    with tf.compat.v1.Session() as sess:
      tf.compat.v1.variables_initializer([signal]).run()
      result = sess.run(fft)
      for i in range(cut*ratio, len(result)-(cut*ratio)+1): #what is the intuition behind this?? 
        result[i] = 0
    ifft = tf.signal.ifft(result)
    with tf.compat.v1.Session() as sess:
      filtered = sess.run(ifft)
  return filtered.real


def highPass(signal, cut, sample_length, sample_rate):
  ''' 
  high pass filter
  args:
    signal = input signal
    cut = filtering bandwidth Hz
    sample_length = total signal length (number of samples)
    sample_rate = sampling rate of the input signal
  return:
    filtered.real = filtered signal
  '''
  ratio = int(sample_length/sample_rate)
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.signal.fft(signal)
    with tf.compat.v1.Session() as sess:
      tf.compat.v1.variables_initializer([signal]).run()
      result = sess.run(fft)
      for i in range(0, (cut*ratio)+1):
        result[i] = 0
      for i in range(len(result)-(cut*ratio), len(result)):
        result[i] = 0
    ifft = tf.signal.ifft(result)
    with tf.compat.v1.Session() as sess:
      filtered = sess.run(ifft)
  return filtered.real


def bandPass(signal, low_cut, high_cut, sample_length, sample_rate):
  ''' 
  band pass filter
  args:
    signal = input signal
    low_cut = filtering bandwidth Hz (lower bound)
    high_cut = filtering bandwidth Hz (upper bound)
    sample_length = total signal length (number of samples)
    sample_rate = sampling rate of the input signal
  return:
    filtered.real = filtered signal
  '''
  ratio = int(sample_length/sample_rate)
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.signal.fft(signal)
    with tf.compat.v1.Session() as sess:
      tf.compat.v1.variables_initializer([signal]).run()
      result = sess.run(fft)
      for i in range(high_cut*ratio, len(result)-(high_cut*ratio)+1):
        result[i] = 0
      for i in range(0, (low_cut*ratio)+1):
        result[i] = 0
      for i in range(len(result)-(low_cut*ratio), len(result)):
        result[i] = 0
    ifft = tf.signal.ifft(result)
    with tf.compat.v1.Session() as sess:
      filtered = sess.run(ifft)
  return filtered.real


def scatterPlot(x, y, gmin, gmax, size=1000):
    ''' 
    make scatter plot image
    args:
        x = channel 1 input list or 1D-array
        y = channel 2 input list or 1D-array
        gmin = minimum value to draw for pixel (1,1)
        gmax = maximum value to draw for pixel (size, size)
        size = image size = (size, size)
    return:
        splot = scatter plot image (numpy.ndarray)
    '''
    if type(x)==np.ndarray and type(y)==np.ndarray:
        pass
    elif type(x)==list and type(y)==list and len(x)*len(y)>0:
        x, y = np.array(x), np.array(y)
    else:
        print('ERROR: invalid type input x,y!\n       x,y must be list or numpy.ndarray')
        return None

    if gmin >= gmax:
        print('ERROR: invalid min, max bounds')
        return None

    splot = np.zeros([size, size])
    # normalize x and y
    x, y = x-gmin, y-gmin
    x, y = x/(gmax-gmin), y/(gmax-gmin)
    x, y = x*(size-1), y*(size-1)
    x, y = x.astype(int), y.astype(int)
    for i in range(len(x)):
        try: splot[x[i], y[i]] += 1
        except: continue
    splot /= splot.max() 
    splot *= 255 
    splot = np.uint8(splot)
    return splot


def toRGB(red, green, blue):
    ''' 
    Create image using R,G,B values of raw data that has passed the band pass filter
    args:
        red = Red value of raw data extracted through bandpass filter
        green = Green value of raw data extracted through bandpass filter
        blue = Blue value of raw data extracted through bandpass filter
    return:
        image = Generated NSP image result
    '''
    if len(red) != len(green) or len(red) != len(blue):
        print('ERROR: different color channel image size')
        return None
    image = np.empty((len(red), len(red), 3), dtype=np.uint8)
    image[:,:,0] = red
    image[:,:,1] = green
    image[:,:,2] = blue
    return image


def processVibSignal(x, y, band1, band2, band3, sample_length, sample_rate):
    ''' 
    R,G,B values are extrated by passing raw data through bandpass filter.
    args:
        x = x axis of raw data
        y = y axis of raw data
        band1 = min, max value of red filter
        band2 = min, max value of green filter
        band3 = min, max value of blue filter
        sample_length = sample_length
        sample_rate = sample_rate
    return:
        toRGB = Pass R,G,B values extracted through band pass filter to toRGB function
    '''
    #Choose the bandpass filters based on Fast-Fourier Transform analysis of time-frequency domain. Choose wisely. 
    x_red = bandPass(x, band1[0], band1[1], sample_length, sample_rate)
    y_red = bandPass(y, band1[0], band1[1], sample_length, sample_rate)
    x_green = bandPass(x, band2[0], band2[1], sample_length, sample_rate)
    y_green = bandPass(y, band2[0], band2[1], sample_length, sample_rate)
    x_blue = bandPass(x, band3[0], band3[1], sample_length, sample_rate)
    y_blue = bandPass(y, band3[0], band3[1], sample_length, sample_rate)
    red = scatterPlot(x_red, y_red, -1, 1, 128)
    green = scatterPlot(x_green, y_green, -1, 1, 128)
    blue = scatterPlot(x_blue, y_blue, -1, 1, 128)

    return toRGB(red, green, blue)


def PLOT_TSNE(data,labels,rms,ae,fpt,test_num):
    ''' 
    Plot feature map using t-SNE and compare the results of RMS, AE, CNN-HS 
    args:
        data = Feature extraction results from test data
        labels = Healthy or Unhealthy results obtained through CNN-HS
        rms = FPT predicted through RMS from test data
        ae = FPT predicted through AE from test data
        fpt = FPT predicted through CNN-HS the test data
        test_num = Order of used test data
    return:
        filtered.real = filtered signal
    '''
    labels = list(labels)
    for j in range(len(labels)):
        temp = labels[j]
        try:
            temp = list(temp)
            labels[j] = temp.index(1)
        except TypeError:
            continue
    #Putting labels for fpt of rms, ae, and CNN-HS
    for i in range(len(labels)):
        if(i>rms):
            labels[i]=2
        elif(i<=rms and i>fpt):
            labels[i]=1
        elif(i<=fpt):
            labels[i]=0

    datalist = [[] for h in range(3)]
    for i in range(len(data)):
        temp = []
        temp.append(data[i])
        temp.append(labels[i])
        datalist[labels[i]].append(temp)
    
    sample_features, sample_labels = [], []
    for label in range(len(datalist)):
        datalist[label] = datalist[label]

    for c in range(len(datalist)):
        for f in range(len(datalist[c])):
            sample_features.append(datalist[c][f][0])
            sample_labels.append(datalist[c][f][1])
    
    print("making t-sne...")

    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
        learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
        min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
        random_state=None, method='barnes_hut', angle=0.5)

    pca = PCA(n_components=100)
    pca_data = pca.fit_transform(sample_features)
    result = tsne.fit_transform(pca_data)

    x = []
    y = []

    for i in range(len(result)):
        x.append(float(result[i][0]))
    for i in range(len(result)):
        y.append(float(result[i][1]))
            
            
    x = np.asarray(x)
    y = np.asarray(y)

    cdict = { 0:'b', 1:'r', 2: 'k'}

    group = sample_labels
    fig, ax = plt.subplots()

    for g in np.unique(group):
        ix = np.where(group == g)
        if g==0:
            state = 'Healthy prediction by all methods'
        elif g==1:
            state = 'Unhealthy prediction by CNN-HS, Healthy by RMS and AE'
        elif g==2:
            state = 'Unhealthy prediction by CNN-HS, RMS, and AE'

        ax.scatter(x[ix], y[ix], c = cdict[g], s = 1)
    

    
    plt.savefig('save_plot/featurmap_'+str(test_num)+'.png')
    print('t-sne image saved')


def PREPARE_DATASET(image,label):
    ''' 
    Process input image and label data according to the given batch size and shuffle num and create dataset
    args:
        image = input NSP image used for training
        label = input label used for training
    return:
        dataset = training dataset
    '''
    image = np.array(image)
    image = np.squeeze(image)
    label = np.array(label)
    label = np.squeeze(label)
    image = tf.cast(image, dtype=tf.float16)
    label = tf.cast(label, dtype =tf.float16)
    image = tf.data.Dataset.from_tensor_slices(image)
    label = tf.data.Dataset.from_tensor_slices(label)
    dataset = tf.data.Dataset.zip((image, label))
    dataset = dataset.shuffle(1000).batch(batch_size=BATCH_SIZE)
    return dataset


def BF_FPT_RELIABILITY(results):
    ''' 
    Calculate Before FPT component
    args:
        results = Healthy or Unhealthy results obtained through CNN-HS 
    return:
        bf_results = List consisting of percent reliability in FPT, number of healthy results before FPT, and number of unhealthy results before FPT
    '''
    reliability = 0
    fpt = FIND_FPT(results)
    bf_results = []

    # If FPT is not detected, all components are return as 0.
    if(fpt == len(results)):
        bf_results = [0,0,0]
        return bf_results

    for i in range(fpt):
        if results[i] == 0:
            reliability +=1
    reliability_result = (reliability / (fpt+1)) * 100
    print("Before fpt reliability : ",reliability_result,"%")
    bf_results = [reliability_result,reliability,fpt-reliability]
    return bf_results


def AF_FPT_RELIABILITY(results):
    ''' 
    Calculate After FPT component
    args:
        results = Healthy or Unhealthy results obtained through CNN-HS
    return:
        af_results = List consisting of percent reliability in FPT, number of healthy results after FPT, and number of unhealthy results after FPT 
    '''
    fpt = FIND_FPT(results)
    reliability = 0
    af_results = []

    # If FPT is not detected, all components are return as 0.
    if(fpt == len(results)):
        af_results = [0,0,0]
        return af_results
    for i in range(fpt,len(results)):
        if results[i] == 1:
            reliability +=1
    reliability_result = (reliability / (len(results)-fpt)) * 100
    print("After fpt reliability : ",reliability_result,"%")

    af_results = [reliability_result,len(results)-fpt-reliability,reliability]
    return af_results


def PREPARE_TRAINING_DATA(data_dir):
    ''' 
    Chnage the bearing's 2channel raw data to NSP image and label according to ratio
    args:
        data_dir = Raw data path used for training
    return:
        train = NSP image used for training
        label = label used for training
    '''
    scaler = preprocessing.MinMaxScaler()
    nsp_image = []
    for filename in os.listdir(data_dir):
        print(filename)
        data = pd.read_csv(os.path.join(data_dir,filename), delimiter=',')
        data = np.array(data)
    
        x = data[:,4] #Get one column data. Customize depending on your data shape
        y = data[:,5] #Get one column data. Customize depending in your data shape
        length = x.shape[0]

        image = processVibSignal(x,y,BAND1, BAND2, BAND3, length,length)
        nsp_image.append(image)

    count=0
    train = []
    label = []
    totalnum = len(nsp_image)
    down_threshhold = int((RATIO/2)*totalnum)
    up_threshhold = int((1-(RATIO/2))*totalnum)

    for image in nsp_image:
        count += 1
        if(count<down_threshhold or count >up_threshhold):
            train.append(image)
            if(count<down_threshhold):
                label.append(0)
            else:
                label.append(1)

    return train,label


def PREPARE_TEST_DATA(data_dir):
    ''' 
    Chnage the bearing's 2channel raw data to NSP image
    args: 
        data_dir = Raw data path used for test
    return:
        test = NSP image used for training
    '''
    test = []
    scaler = preprocessing.MinMaxScaler()

    for filename in os.listdir(data_dir):
        print(filename)
        data = pd.read_csv(os.path.join(data_dir,filename), delimiter=',')
        data = np.array(data)
    
        x = data[:,4] #Get one column data. Customize depending on your data shape
        y = data[:,5] #Get one column data. Customize depending in your data shape
        length = x.shape[0]

        image = processVibSignal(x,y,BAND1, BAND2, BAND3, length,length)
        test.append(image)
    test = np.array(test)
    test = test.astype('float16')

    return test
        

def PLOT_RESULT(results,test_num):
    ''' 
    Plot the prediction result and FPT of CNN-HS model
    args:
        results = Healthy or Unhealthy results obtained through CNN-HS
        test_num = Order of used test data
    '''
    plt.cla()
    plt.clf()
    fpt = FIND_FPT(results)
    plt.plot(range(len(results)),results, color='k')
    plt.axvline(x=fpt, color = 'r', linewidth = 3, linestyle = '--', label='FPT')
    plt.ylabel('Output',fontsize=22)
    plt.xlabel('Time',fontsize=20)
    plt.ylim([-0.1,1.1])
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig('save_plot/TEST_'+str(test_num)+'.png')


def FIND_FPT(results):
    ''' 
    Find first predicting time using CNN-HS model results 
    args:
        results = Healthy or Unhealthy results obtained through CNN-HS
    return:
        fpt = The fastest spot observed unhealthy 3 times in results
    '''
    count=0
    fpt = len(results)
    for i in range(len(results)):
        for j in range(3):
            if(i+j == len(results)-1):
                    print("\n\ncould not find FPT\n\n")
                    fpt = len(results)
                    return fpt
            if (results[i+j]==1):
                count+=1
                if(count==3):
                    fpt = i
                    return fpt
        count = 0   
    return fpt



def HS_DETERMINE(predict):
    ''' 
    Determine healthy or unhealthy using predict results.
    args:
        predict = Result of predicting NSP image using CNN-HS
    return:
        results = Healthy unhealthy judgment result using predict
    '''
    results = []
    for i in range(len(predict)):
        if predict[i][0] > predict[i][1]:
            results.append(0)
        else:
            results.append(1)
    return results


def F1_SCORE(bf_results,af_results):
    ''' 
    Calculate f1 score using results of After FPT and Before FPT
    args:
        bf_results = List consisting of percent reliability in FPT, number of healthy results before FPT, and number of unhealthy results before FPT
        af_results = List consisting of percent reliability in FPT, number of healthy results after FPT, and number of unhealthy results after FPT 
    return:
        f1_score = Calculated f1 score value
    '''
    precision = af_results[2] / (bf_results[2] + af_results[2])
    recall =  af_results[2] / (af_results[1] + af_results[2])
    f1_score = 2*precision*recall / (precision+recall)
    print("f1_score : ",f1_score)
    return f1_score

