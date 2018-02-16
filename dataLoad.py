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
        self.fileReader = []
        self.lock = threading.Lock()
        self.inputNames = self.config.input_names.split(',')
        self.outputNames = config.output_names.split(',')
        self.varAllList = self.inputNames + self.outputNames
        #if 'SPDT' in self.varAllList:
        #    if 'TPHYSTND' in self.varAllList:
        #        # tendency due to everything but convection
        #        self.inputNames += ['dTdt_nonSP']
        #if 'SPDQ' in self.varAllList:
        #    if 'PHQ' in self.varAllList:
        #        # tendency due to everything but convection
        #        self.inputNames += ['dQdt_nonSP']
        print('self.varAllList', self.varAllList)
        self.varNameSplit = -len(self.outputNames)
        self.rawFileBase = rawFileBase
        if self.config.normalizeInoutputs:
            print(mean_file)
            self.mean_data = nc.Dataset(mean_file, mode='r') 
            #self.mean_data = h5py.File(mean_file, 'r')  # needed in UNIX as header is not correctly read
            print(std_file)
            self.std_data = nc.Dataset(std_file, mode='r') 
            #self.std_data = h5py.File(std_file, 'r') 
        self.reload() 

    def reload(self):
        raw_data_train_path = trainingDataDirRaw+'*.nc'
        print(Fore.YELLOW, 'raw_data_train_path', raw_data_train_path, Style.RESET_ALL)
        # read addresses and labels from the 'train' folder
        self.rawFiles = {}
        self.rawDates = []
        self.varDim = {}
        for fn in  sorted(glob.glob(raw_data_train_path)):
            date = fn.split('.')[-2]
            self.rawDates += [date]
            self.rawFiles[date] = fn
        print(self.rawDates[:3])
        print('last raw file:', fn)
        with nc.Dataset(fn, mode='r') as aqua_rg:
            self.n_tim = aqua_rg.dimensions['time'].size
            if self.config.nlevs_imposed==0:
                self.n_lev = aqua_rg.dimensions['lev'].size
            else:
                self.n_lev = self.config.nlevs_imposed
            self.n_lat = aqua_rg.dimensions['lat'].size
            self.n_lon = aqua_rg.dimensions['lon'].size
            print(aqua_rg)
            for k in aqua_rg.variables.keys():
                self.varDim[k] = len(aqua_rg[k].shape)
                if len(aqua_rg[k].shape) > 2:
                    print(Fore.YELLOW)
                print(fn+': ', k, aqua_rg[k].shape, Style.RESET_ALL)
            print('n_tim =', self.n_tim, " = ", aqua_rg.variables['time'][:3],"...",aqua_rg.variables['time'][-3:])
            print('n_lev =', self.n_lev, " = ", aqua_rg.variables['lev'][:3],"...",aqua_rg.variables['lev'][-3:])
            print('n_lat =', self.n_lat, " = ", aqua_rg.variables['lat'][:3],"...",aqua_rg.variables['lat'][-3:])
            print('n_lon =', self.n_lon, " = ", aqua_rg.variables['lon'][:3],"...",aqua_rg.variables['lon'][-3:])
            # if flattened, the split is not the index of the first output name, but the index of the first output once flattened
            if not True:#:
                self.varNameSplit = self.accessTimeData(aqua_rg, self.inputNames, 0, doLog=True).shape[0]
            print('self.varNameSplit', self.varNameSplit)
            sampX, sampY = self.prepareData(aqua_rg, 0, doLog=True)
            print('sampX =', sampX.shape)
            print('sampY =', sampY.shape)

        try:
            for i in range(len(self.fileReader)):
                self.fileReader[i].close()
        except:
            pass
        print("batchSize = ", self.config.batch_size)

        self.Nsamples = len(self.rawDates) * self.n_tim * self.n_lon * self.n_lat
        self.NumBatch = self.Nsamples // self.config.batch_size

        self.Xshape = list(sampX.shape)
        self.Yshape = list(sampY.shape)
        print('Xshape', self.Xshape)
        print('Yshape', self.Yshape)

        ## this deals with tf records and using them by date for training/validation
        if not self.rawFileBase:
            tfRecordsFolderName = '/'.join(self.recordFileName(trainingDataDirTFRecords+date+'/t{0:02d}').split('/')[:-1])
            folders = sorted(glob.glob(trainingDataDirTFRecords+'*'))
            foldersSplit = int(len(folders) * self.config.frac_train + 0.5)
            print(folders[:3], len(folders), folders[-3:])
            folders = folders[:foldersSplit] if self.config.is_train else folders[foldersSplit:]
            print(Fore.RED, 'days', folders[0], '-->', folders[-1], Style.RESET_ALL)#[fn.split('/')[-1] for fn in folders], Style.RESET_ALL)
            self.tfRecordsFiles = []
            for fn in folders:
                self.tfRecordsFiles += glob.glob(fn+"/*" + '_c' + ".tfrecords")
            self.tfRecordsFiles = sorted(self.tfRecordsFiles)
            print("tfRecordsFiles", len(self.tfRecordsFiles))

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
        if varname == "SPDQ" or varname == "PHQ":
            return arr*2.5e6/1000.
        return arr 
    
    # normalize data based on mean and std.nc files
    def normalizeInoutputs(self, varname, arr):
        meandata= np.array(self.mean_data.variables[varname])
        stddata = np.array(self.std_data.variables[varname])
        arr     = (arr - meandata)/stddata
#        test    = np.sum(np.where(mean>50000))
#        test2   = np.sum(np.where(mean>50000))
#        if(test+test2>0):
#            print("problem")
        return arr
    
    def accessTimeData(self, fileReader, names, iTim, doLog=False):
        inputs = []
        
        for k in names:
            if doLog: 
                print('accessTimeData', 'varDim[k]=', self.varDim[k], 'fileReader[k]=', fileReader[k].shape)
            if self.varDim[k] == 4: 
                arr = fileReader[k][iTim]
            elif self.varDim[k] == 3:
                arr = fileReader[k][iTim][None]
            elif self.varDim[k] == 2:
                arr = fileReader[k][None] 
            elif self.varDim[k] == 1: # latitude only
                # need to transform in single as latitude is in double
                arr = fileReader[k]
                arr = np.swapaxes(np.tile(arr, (1,self.n_lon,1)),1,2)# repeat lat to trasnform into matrix
                arr = arr.astype('float32') # impose float 32 like other varaiables
            if doLog: 
                print(k, '----> arr=', arr.shape)
            # noamlize data firs, better for convergences
            if self.config.normalizeInoutputs:
                arr = self.normalizeInoutputs(k, arr)
            else: # does not allow double nmormalizations
                if self.config.convert_units:
                    arr = self.convertUnits(k, arr)
            if self.varDim[k] == 4: # only keep n top pressure levels        
                arr = arr[(arr.shape[0]-self.n_lev):(arr.shape[0]),:,:] # select just n levels
            if doLog: 
                print(k, '--------> arr=', arr.shape)
            if arr.shape[0] == 1:
                arr = np.tile(arr, (self.n_lev,1,1))
            if doLog: 
                print(k, '------------> arr=', arr.shape)
            inputs += [arr]

        inX = np.stack(inputs, axis=0)
        if doLog: 
            print('accessTimeData ', names, inX.shape)
        return inX


    def prepareData(self, fileReader, iTim, doLog=False):
        samp = self.accessTimeData(fileReader, self.varAllList, iTim, doLog)
        return np.split(samp, [self.varNameSplit])

    def get_inputs(self):
        return self.get_record_inputs(self.config.is_train, self.config.batch_size, self.config.epoch)

    def recordFileName(self, filename):
        return filename + '_c' + '.tfrecords' # address to save the TFRecords file into

    def makeTfRecordsDate(self, date):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def _floats_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))
        filenameBase = self.recordFileName(trainingDataDirTFRecords+date+'/t{0:02d}')
        folderName = '/'.join(filenameBase.split('/')[:-1])
        #print(filenameBase)
        #print(folderName)
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        shards = self.n_tim
        sampBar = tqdm(range(shards), leave=False) 
        with nc.Dataset(self.rawFiles[date], mode='r') as aqua_rg:
            for iTim in sampBar:
                # open the TFRecords file
                filename = filenameBase.format(iTim)
                #print('opening the TFRecords file', filename)
                writer = tf.python_io.TFRecordWriter(filename)
                sampBar.set_description(folderName)
                sX, sY = self.prepareData(aqua_rg, iTim)
                # Create a feature
                #print(sX.shape, sX.dtype)
                #print(sY.shape, sY.dtype)
                if True:
                    if True:
                        feature = {'X': _bytes_feature(tf.compat.as_bytes(sX.tostring())),
                                   'Y': _bytes_feature(tf.compat.as_bytes(sY.tostring()))
                                   }
#                for iLat in range(sX.shape[-2]):
#                    for iLon in range(sX.shape[-1]):
#                        feature = {'X': _bytes_feature(tf.compat.as_bytes(sX[:,:,iLat,iLon].tostring())),
#                               'Y': _bytes_feature(tf.compat.as_bytes(sY[:,:,iLat,iLon].tostring()))
#                               }
                        # Create an example protocol buffer
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # Serialize to string and write on the file
                        writer.write(example.SerializeToString())
                writer.close()
                sys.stdout.flush()

    def makeTfRecords(self, n_threads=4):
        """ Start background threads to feed queue """
        threads = []
        print("starting %d data threads for making records" % n_threads)
#        for k in range(len(self.rawDates)):
#            t = threading.Thread(target=self.makeTfRecordsDate, args=(self.rawDates[k]))
#            t.daemon = True # thread will close when parent quits
#            t.start()
#            threads.append(t)
        daysBar = tqdm(range(len(self.rawDates)))
        for k in daysBar:
            date = self.rawDates[k]
            self.makeTfRecordsDate(date)

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
            features={
            'X': tf.FixedLenFeature([], tf.string),
            'Y': tf.FixedLenFeature([], tf.string)
        })
        X = tf.decode_raw(features['X'], tf.float32)
        Y = tf.decode_raw(features['Y'], tf.float32)
        #print('read_and_decode X', X)
        X.set_shape(np.prod(self.Xshape))
        Y.set_shape(np.prod(self.Yshape))
        #print('read_and_decode X', X)
        X = tf.reshape(X, self.Xshape)
        Y = tf.reshape(Y, self.Yshape)
        print('read_and_decode X', X)
        print('read_and_decode Y', Y)
        return X, Y

    def get_record_inputs(self, train, batch_size, num_epochs):
        """Reads input data num_epochs times.
        Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
        """
        if not num_epochs: num_epochs = None
        with tf.name_scope('dequeue'):
            filename_queue = tf.train.string_input_producer(self.tfRecordsFiles, num_epochs=num_epochs, shuffle=self.config.randomize)
            print('filename_queue', filename_queue)
            X, Y = self.read_and_decode(filename_queue)
            X = tf.transpose(tf.reshape(X, self.Xshape[:2]+[-1]), [2,0,1])
            Y = tf.transpose(tf.reshape(Y, self.Yshape[:2]+[-1]), [2,0,1])
            X = tf.expand_dims(X, -1)
            Y = tf.expand_dims(Y, -1)
            X = X[:,:,-self.n_lev:,:][:,:,::-1]
            Y = Y[:,:,-self.n_lev:,:][:,:,::-1]
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            self.capacityTrain = 8192 * 32
            if self.config.randomize:
                b_X, b_Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, num_threads=2,
                                                    enqueue_many=True,
                                                    capacity=self.capacityTrain,
                                                    min_after_dequeue=self.capacityTrain // 2)
            else:
                b_X, b_Y = tf.train.batch([X, Y], batch_size=batch_size, num_threads=2,
                                                    enqueue_many=True,
                                            capacity=self.capacityTrain)
            print('self.capacityTrain', self.capacityTrain)
        return b_X, b_Y

if __name__ == "__main__":
    config, unparsed = get_config()
    print(Fore.GREEN, 'config\n', config)
    print(Fore.RED, 'unparsed\n', unparsed)
    print(Style.RESET_ALL)
    if unparsed:
        assert(False)
    dh = DataLoader(trainingDataDir, config, raw_file_base)
    dh.makeTfRecords()
