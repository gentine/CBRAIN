import numpy as np
import shutil, time, math, itertools, os
import h5netcdf as h5py
import netCDF4 as nc
from tqdm import tqdm
import tensorflow as tf
import threading
import random
from colorama import Fore, Back, Style
from config import get_config
import sys
from folderDefs import *
import glob

class DataLoader:
    def __init__(self, folderPath, config, rawFileBase=''):
        self.config = config
        self.nSampleFetching = 1024
        self.varnameList = config.dataset.split(',')
        self.fileReader = []
        self.lock = threading.Lock()
        self.inputNameList = self.config.input_names.split(',')
        self.rawFileBase = rawFileBase
        self.reload()

    def reload(self):
        shuffle_data = False  # shuffle the addresses before saving
        cat_dog_train_path = trainingDataDirRaw+'*.nc'
        # read addresses and labels from the 'train' folder
        self.rawFiles = {}
        self.rawDates = []
        for fn in  glob.glob(cat_dog_train_path):
            date = fn.split('.')[-2]
            self.rawDates += [date]
            self.rawFiles[date] = fn
        print(self.rawFiles)
        print('last raw file:', fn)
        with nc.Dataset(fn, mode='r') as aqua_rg:
            self.n_tim = aqua_rg.dimensions['time'].size
            self.n_lev = aqua_rg.dimensions['lev'].size
            self.n_lat = aqua_rg.dimensions['lat'].size
            self.n_lon = aqua_rg.dimensions['lon'].size
            print(aqua_rg)
            for k in aqua_rg.variables.keys():
                print(fn+': ', k, aqua_rg[k].shape)
            print('n_tim =', self.n_tim)
            print('n_lev =', self.n_lev)
            print('n_lat =', self.n_lat)
            print('n_lon =', self.n_lon)
            print(aqua_rg.variables['lev'][:])
            sampX, sampY = self.accessData(0, self.nSampleFetching, aqua_rg)

        try:
            for i in range(len(self.fileReader)):
                self.fileReader[i].close()
        except:
            pass
        print("batchSize = ", self.config.batch_size)

#        with h5py.File(nc_file, mode='r') as fh:
#            for k in fh.keys():
#                print('nc_file: ', k, fh[k].shape)
#            self.Nsamples = fh[k].shape[0]
#            print('Nsamples =', self.Nsamples)
#            self.Nlevels      = self.mean['QAP'].shape[1]
#            print('Nlevels = ', self.Nlevels)
#             sampX, sampY = self.accessData(0, self.nSampleFetching, fh)
#            self.n_input = sampX.shape[1] # number of inputs 
#            self.n_output = sampY.shape[1] # number of outputs 
#             print('sampX = ', sampX.shape)
#             print('sampY = ', sampY.shape)
#            print('n_input = ', self.n_input)
#            print('n_output = ', self.n_output)
#
        self.Nsamples = self.n_tim * len(self.rawDates)
        self.NumBatch = self.Nsamples // self.config.batch_size
        self.NumBatchTrain = int(self.Nsamples * self.config.frac_train) // self.config.batch_size
        self.indexValidation = self.NumBatchTrain * self.config.batch_size
        self.NumBatchValid = int(self.Nsamples * (1.0 - self.config.frac_train)) // self.config.batch_size
        print('NumBatch=', self.NumBatch)
        print('NumBatchTrain=', self.NumBatchTrain)
        print('indexValidation=', self.indexValidation)
        print('NumBatchValid=', self.NumBatchValid)

        self.samplesTrain = range(0, self.indexValidation, self.nSampleFetching)
        self.randSamplesTrain = list(self.samplesTrain)
        if self.config.randomize:
            random.shuffle(self.randSamplesTrain)
        self.samplesValid = range(self.indexValidation, self.Nsamples, self.nSampleFetching)
        self.randSamplesValid = list(self.samplesValid)
        if self.config.randomize:
            random.shuffle(self.randSamplesValid)
        self.numFetchesTrain = len(self.randSamplesTrain)
        self.numFetchesValid = len(self.randSamplesValid)
        print('randSamplesTrain', self.randSamplesTrain[:16], self.numFetchesTrain)
        print('randSamplesValid', self.randSamplesValid[:16], self.numFetchesValid)
        self.posTrain = 0
        self.posValid = 0

        self.Xshape = list(sampX.shape[1:])
        self.Yshape = list(sampY.shape[1:])
        print('Xshape', self.Xshape)
        print('Yshape', self.Yshape)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for i in range(len(self.fileReader)):
                self.fileReader[i].close()
        except:
            pass

    def convertUnits(self, varname, arr):
        """Make sure SPDQ and SPDT have comparable units"""
        if varname == "SPDT":
            return arr*1000
        if varname == "SPDQ":
            return arr*2.5e6
        return arr

    def readDatasetY(self, s, l, fileReader):
        data = []
        for k in self.varnameList:
            try:
                arr = fileReader[k][s:s+l,:,:,:].transpose([0,2,3,1])
            except:
                arr = fileReader[k][s:s+l][:,None,:,:].transpose([0,2,3,1])
            if self.config.convert_units:
                arr = self.convertUnits(k, arr)
            data += [arr]
        if self.config.convo:
            y_data = np.stack(data, axis=-1)
        else:
            y_data = np.concatenate(data, axis=-1) #[b,cc]
        
        return y_data

    def accessData(self, s, l, fileReader):
        inputs = []
        for k in self.inputNameList:
            try:
                arr = fileReader[k][s:s+l,:,:,:].transpose([0,2,3,1])
            except:
                arr = fileReader[k][s:s+l][:,None,:,:].transpose([0,2,3,1])
            print(k, arr.shape)

            if self.config.convo:
                if arr.shape[-1] == 1:
                    arr = np.tile(arr, (1,1,1,self.n_lev))
                #arr = arr[:,:,:,:,None] #[t,lat,lon,lev,1]
                print(k, arr.shape)
            inputs += [arr]
        # input output data
        if self.config.convo:
            inX = np.stack(inputs, axis=-1) #[b,lat,lon,lev,chan]
        else: # make a soup of numbers
            inX = np.concatenate(inputs, axis=-1) #[b,lon,lat,cc]
        y_data = self.readDatasetY(s, l, fileReader)

        # flattens t,lon,lat
        inX = inX.reshape((-1,)+inX.shape[-2:])
        y_data = y_data.reshape((-1,)+y_data.shape[-2:])
        return inX, y_data

    def sampleTrain(self, ithFileReader):
#        self.lock.acquire()
        s = self.randSamplesTrain[self.posTrain]
        #print(ithFileReader, self.posTrain, s)
        self.posTrain += 1
        self.posTrain %= self.numFetchesTrain
#        self.lock.release()
        x,y = self.accessData(s, self.nSampleFetching, self.fileReader[ithFileReader])
        return x,y

    def sampleValid(self, ithFileReader):
        s = self.randSamplesValid[self.posValid]
        self.posValid += 1
        self.posValid %= self.numFetchesValid
        x,y = self.accessData(s, self.nSampleFetching, self.fileReader[ithFileReader])
        return x,y

    def data_iterator(self, ithFileReader):
        """ A simple data iterator """
        print('data_iterator', ithFileReader, threading.current_thread())
        while True:
            sampX, sampY = self.sampleTrain(ithFileReader) if self.config.is_train else self.sampleValid(ithFileReader)
            yield sampX, sampY

    def prepareQueue(self):
        with tf.name_scope('prepareQueue'):
            self.dataX = tf.placeholder(dtype=tf.float32, shape=[None]+self.Xshape)
            self.dataY = tf.placeholder(dtype=tf.float32, shape=[None]+self.Yshape)

            self.capacityTrain = max(self.nSampleFetching * 32, self.config.batch_size * 8) if self.config.is_train else self.config.batch_size
            if self.config.randomize:
                self.queue = tf.RandomShuffleQueue(shapes=[self.Xshape, self.Yshape],
                                               dtypes=[tf.float32, tf.float32],
                                               capacity=self.capacityTrain,
                                               min_after_dequeue=self.capacityTrain // 2
                                               )
            else:
                self.queue = tf.FIFOQueue(shapes=[self.Xshape, self.Yshape],
                                               dtypes=[tf.float32, tf.float32],
                                               capacity=self.capacityTrain
                                               )
            self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])
            self.size_op = self.queue.size()

    def get_inputs(self):
        with tf.name_scope('dequeue'):
            train0Valid1 = tf.placeholder_with_default(1, [], name='train0Valid1')
            b_X, b_Y = self.queue.dequeue_many(self.config.batch_size)
            print("b_X",b_X.get_shape(), "b_Y",b_Y.get_shape())
            return b_X, b_Y

    def thread_main(self, sess, ithFileReader):
        print('thread_main', ithFileReader, threading.current_thread())
        while len(self.fileReader) <= ithFileReader + 1:
            self.fileReader += [h5py.File(nc_file, mode='r')]
        for dtX, dtY in self.data_iterator(ithFileReader):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dtX, self.dataY:dtY})

    def start_threads(self, sess, n_threads=4):
        """ Start background threads to feed queue """
        threads = []
        print("starting %d data threads for training" % n_threads)
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,0,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        # Make sure the queueu is filled with some examples (n = 500)
        num_samples_in_queue = 0
        while num_samples_in_queue < self.capacityTrain:
            num_samples_in_queue = sess.run(self.size_op)
            print("Initializing queue, current size = %i/%i" % (num_samples_in_queue, self.capacityTrain))
            time.sleep(2)
        return threads
