from __future__ import print_function

import tensorflow as tf
import keras.backend as K


def makeLossesPerLevel(y, pred, names, lossfct):
    print('makeLossesPerLevel')
    print('y:', y)
    print('pred:', pred)
    numChanOut = y.get_shape().as_list()[1]
    assert(numChanOut == len(names))
    print('numChanOut:', numChanOut)
    print(names)
    lossDict = {}   
    with tf.name_scope('lossPerLevel'):
        error = y - pred
        sqrLosses = tf.square(error, name='sqrLosses')
        absLosses = tf.abs(error, name='absLosses')
        loglosses = tf.divide(tf.log(absLosses+1e-9), tf.log(10.0), name='loglosses')
        batchAvgY = tf.reduce_mean(y, axis=0, keep_dims=True, name='batchAvgY')
        batchAvgPred = tf.reduce_mean(pred, axis=0, keep_dims=True, name='batchAvgPred')
        for iOut in range(len(names)):
            outName = names[iOut]
            lossDict['sqrLossesPerLev'+'/'+outName] = tf.reduce_mean(sqrLosses[:,iOut,:], axis=0)
            lossDict['absLossesPerLev'+'/'+outName] = tf.reduce_mean(absLosses[:,iOut,:], axis=0)
            lossDict['loglossesPerLev'+'/'+outName] = tf.reduce_mean(loglosses[:,iOut,:], axis=0)
            lossDict['meanYPerLev'+'/'+outName]     = tf.reduce_mean(batchAvgY[:,iOut,:], axis=0)
            lossDict['meanPredPerLev'+'/'+outName]  = tf.reduce_mean(batchAvgPred[:,iOut,:], axis=0)
            lossDict['totErrPerLev'+'/'+outName]    = tf.reduce_mean(tf.square(y[:,iOut,:] - batchAvgY[:,iOut,:]), axis=0)
            lossDict['R2PerLev'+'/'+outName]        = tf.nn.relu(1 - lossDict['sqrLossesPerLev'+'/'+outName] / (lossDict['totErrPerLev'+'/'+outName] + 1e-15))

    with tf.name_scope('lossAvgLevel'):
        keys = list(lossDict.keys())
        for n in keys:
            lossDict[n.replace('PerLev', 'AvgLev')] = tf.reduce_mean(lossDict[n], axis=-1)

    with tf.name_scope('loss'):
        lossDict['RMSE'] = tf.sqrt(tf.reduce_mean(sqrLosses))
        lossDict['mse'] = tf.reduce_mean(sqrLosses)
        lossDict['logloss'] = tf.reduce_mean(loglosses)
        lossDict['absloss'] = tf.reduce_mean(absLosses)
        lossDict['R2'] = tf.nn.relu(1 - tf.reduce_mean(sqrLosses) / (tf.reduce_mean(tf.square(y[:,:,:] - batchAvgY[:,:,:])) + 1e-15))

        # choose cost function
        if lossfct=="logloss":
            lossDict['loss'] = lossDict[lossfct]
        if lossfct=="abs":
            lossDict['loss'] = lossDict['absloss']
        if lossfct=="Rsquared":
            lossDict['loss'] = -lossDict['R2']
        if lossfct=="mse":
            lossDict['loss'] = lossDict['mse']
        if lossfct=="RMSE":
            lossDict['loss'] = lossDict['RMSE']
        lossDict['loss'] = tf.identity(lossDict['loss'], name="loss")

    for n in lossDict.keys():
        print(n, lossDict[n])
    return lossDict