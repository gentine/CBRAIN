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
            lossDict['sqrLossesPerLev'+'/'+outName] = tf.reduce_mean(sqrLosses[:,iOut,:], axis=0, name='sqrLossesPerLev'+'/'+outName)
            lossDict['absLossesPerLev'+'/'+outName] = tf.reduce_mean(absLosses[:,iOut,:], axis=0, name='absLossesPerLev'+'/'+outName)
            lossDict['loglossesPerLev'+'/'+outName] = tf.reduce_mean(loglosses[:,iOut,:], axis=0, name='loglossesPerLev'+'/'+outName)
            lossDict['meanYPerLev'+'/'+outName]     = tf.reduce_mean(batchAvgY[:,iOut,:], axis=0, name='meanYPerLev'+'/'+outName)
            lossDict['meanPredPerLev'+'/'+outName]  = tf.reduce_mean(batchAvgPred[:,iOut,:], axis=0, name='meanPredPerLev'+'/'+outName)
            lossDict['totErrPerLev'+'/'+outName]    = tf.reduce_mean(tf.square(y[:,iOut,:] - batchAvgY[:,iOut,:]), axis=0, name='totErrPerLev'+'/'+outName)
            lossDict['R2PerLev'+'/'+outName]        = tf.identity(1 - lossDict['sqrLossesPerLev'+'/'+outName] / (lossDict['totErrPerLev'+'/'+outName] + 1e-15), name='R2PerLev'+'/'+outName)

    with tf.name_scope('lossAvgLevel'):
        keys = list(lossDict.keys())
        for n in keys:
            lossDict[n.replace('PerLev', 'AvgLev')] = tf.reduce_mean(lossDict[n], axis=-1, name=n.replace('PerLev', 'AvgLev'))

    with tf.name_scope('loss'):
        lossDict['RMSE'] = tf.sqrt(tf.reduce_mean(sqrLosses), name='RMSE')
        lossDict['mse'] = tf.reduce_mean(sqrLosses, name='mse')
        lossDict['logloss'] = tf.reduce_mean(loglosses, name='logloss')
        lossDict['absloss'] = tf.reduce_mean(absLosses, name='absloss')
        lossDict['R2'] = tf.identity(1 - tf.reduce_mean(sqrLosses) / (tf.reduce_mean(tf.square(y[:,:,:] - batchAvgY[:,:,:])) + 1e-15), name='R2')

        # choose cost function
        if lossfct=="logloss":
            lossDict['loss'] = tf.identity(lossDict[lossfct], name="loss")
        if lossfct=="abs":
            lossDict['loss'] = tf.identity(lossDict['absloss'], name="loss")
        if lossfct=="Rsquared":
            lossDict['loss'] = tf.identity(-lossDict['R2'], name="loss")
        if lossfct=="mse":
            lossDict['loss'] = tf.identity(lossDict['mse'], name="loss")
        if lossfct=="RMSE":
            lossDict['loss'] = tf.identity(lossDict['RMSE'], name="loss")

    for n in lossDict.keys():
        print(n, lossDict[n])
    return lossDict