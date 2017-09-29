import numpy as np
import shutil, time, math, itertools, os
import h5py
#import netCDF4 as nc 
#from netCDF4 import Dataset
from tqdm import tqdm
import tensorflow as tf
import threading
import random
from colorama import Fore, Back, Style
import sys
from folderDefs import *

class DataLoader:
    def __init__(self, folderPath, config):
        self.config = config
        self.batchSize = config.batch_size
        self.nSampleFetching = 1024
        self.varnameList = config.dataset.split(',')
        self.fileReader = []
        self.lock = threading.Lock()
        self.inputNameList = self.config.input_names.split(',')
        self.reload()

    def reload(self, finishedEpoch = 0):
        # need to retrieve mean and standard deviation of the full dataset first
        print("Reading Netcdfs mean and std for Normalization")
        self.mean = {}
        self.std = {}
        self.max = {}
        with h5py.File(nc_mean_file, mode='r') as fh:
            for k in fh.keys():
                try:
                    self.mean[k] = fh[k][None,:]
                except:
                    self.mean[k] = np.array(fh[k])[None]
                print('nc_mean_file: ', k, self.mean[k].shape)#, self.mean[k])
        with h5py.File(nc_std_file, mode='r') as fh:
            for k in fh.keys():
                try:
                    self.std[k] = fh[k][None,:]
                except:
                    self.std[k] = np.array(fh[k])[None]
                print('nc_std_file: ', k, self.std[k].shape)#, self.std[k])
 
        #with h5py.File(nc_max_file, mode='r') as fh: # normalize outputs to be between -1 and 1
        #    for k in fh.keys():
        #        try:
        #            self.max[k] = fh[k][None,:]
        #        except:
        #            self.max[k] = np.array(fh[k])[None]
        #        print('nc_max_file: ', k, self.max[k].shape)#, self.max[k])
        
        print("End Reading Netcdfs for Normalization")
        try:
            for i in range(len(self.fileReader)):
                self.fileReader[i].close()
        except:
            pass
        print("batchSize = ", self.batchSize)

        with h5py.File(nc_file, mode='r') as fh:
            for k in fh.keys():
                print('nc_file: ', k, fh[k].shape)
            self.Nsamples = fh[k].shape[0]
            print('Nsamples =', self.Nsamples)
            self.Nlevels      = self.mean['QAP'].shape[1]
            print('Nlevels = ', self.Nlevels)
            sampX, sampY = self.accessData(0, self.nSampleFetching, fh)
            self.n_input = sampX.shape[1] # number of inputs 
            self.n_output = sampY.shape[1] # number of outputs 
            print('sampX = ', sampX.shape)
            print('sampY = ', sampY.shape)
            print('n_input = ', self.n_input)
            print('n_output = ', self.n_output)

        self.NumBatch = self.Nsamples // self.config.batch_size
        self.NumBatchTrain = int(self.Nsamples * self.config.frac_train) // self.batchSize
        self.indexValidation = self.NumBatchTrain * self.batchSize
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

    def readDatasetY(self, s, l, fileReader, varnameList, convo):
        data = []
        manyvar = len(varnameList) > 1
        if convo:
            for k in varnameList:
                try:
                    arr = fileReader[k][:,s:s+l].T[:,:,None,None]# [b,h,1,c=1]
                except:
                    arr = np.array(fileReader[k][s:s+l])[None,:].T[:,:,None,None]
                if manyvar:
                    arr = self.convertUnits(k, arr)
                data += [arr]
            y_data = np.concatenate(data, axis=3) #[b,z,1,c]
        else:
            for k in varnameList:
                try:
                    arr = fileReader[k][:,s:s+l].T
                except:
                    arr = np.array(fileReader[k][s:s+l])[None,:].T
                if manyvar:
                    arr = self.convertUnits(k, arr)
                data += [arr]
            y_data = np.concatenate(data, axis=-1) #[b,cc]
        
        return y_data

    # =========================================================================
    # Making dummy data here for network test; linear transform of OMEGA,
    # +/- a random perturbation; if the network can predict this
    # relationship, then at least the model works; if not, there is a bug
    # =========================================================================
    def makeDummyData(self, x_data, is_convo):
        y_data = x_data * 1e4
        if is_convo:
            rows,cols,tmp1,tmp2 = y_data.shape
            for i in range(rows):
                for j in range(cols):
                    y_data[i,j,0,0] = y_data[i,j,0,0] + random.uniform(-1e3,1e3)
        else:
            rows,cols = y_data.shape
            for i in range(rows):
                for j in range(cols):
                    y_data[i,j] = y_data[i,j] + random.uniform(-1e3,1e3)
        return y_data

    def accessData(self, s, l, fileReader):
        inputs = []
        for k in self.inputNameList:#fileReader.keys():
            #print('nc_file: ', k, fileReader[k].shape)
            try:
                arr = fileReader[k][:,s:s+l].T
            except:
                arr = np.array(fileReader[k][s:s+l])[None,:].T
            # normalize data
            if self.config.normalize:
                arr -= self.mean[k]
                arr /= self.std[k]
            if s == 0:
                print('nc_file: ', k, arr.shape)

            if self.config.convo:
                if arr.shape[-1] == 1:
                    arr = np.tile(arr, (1,self.Nlevels))
                arr = arr[:,:,None] #[b,z,1]
                #print('nc_file: ', k, arr.shape)
            inputs += [arr]
        # input output data
        if self.config.convo:
            inX = np.stack(inputs, axis=-1) #[b,z,1,c]
        else: # make a soup of numbers
            inX = np.concatenate(inputs, axis=-1) #[b,cc]
        #y_data = self.readDatasetY(s, l, fileReader, self.varnameList, self.config.convo) # real data routine
        y_data = self.makeDummyData(inX, self.config.convo) # dummy data routine

        if False:
            # remove any rows with NaNs
            tmp = inX
            print('tmp.shape', tmp.shape)
            inX = np.empty((0,tmp.shape[1]), float)
            print('inX.shape', inX.shape)
            y_data_good = np.empty((0,1), float)
            print('y_data_good.shape', y_data_good.shape)
            i = 0
            for row in tmp:
                test = np.isnan(row).any()
                if not test:
                    inX = np.vstack([inX, row])
                    y_data_good = np.vstack([y_data_good, y_data[i]])
                i += 1
            y_data = y_data_good

        if s == 0:
            print('inX.shape', inX.shape)
            print('y_data.shape', y_data.shape)

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

            self.capacityTrain = max(self.nSampleFetching * 32, self.batchSize * 8) if self.config.is_train else self.batchSize
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
            b_X, b_Y = self.queue.dequeue_many(self.batchSize)
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

