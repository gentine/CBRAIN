from __future__ import print_function

import tensorflow as tf
import keras.backend as K


def makeLossesPerVar(y, pred, names, lossfct):
    print('makeLossesPerVar')
    print('y:', y)
    print('pred:', pred)
    numChanOut = y.get_shape().as_list()[1]
    assert(numChanOut == len(names))
    print('numChanOut:', numChanOut)
    print(names)
    lossDict = {}   
    with tf.name_scope('PerVar'):
        error = y - pred
        sqrLosses = tf.square(error, name='sqrLosses')
        absLosses = tf.abs(error, name='absLosses')
        loglosses = tf.divide(tf.log(absLosses+1e-15), tf.log(10.0), name='loglosses')
        batchAvgY = tf.reduce_mean(y, axis=0, keep_dims=True, name='batchAvgY')
        batchAvgPred = tf.reduce_mean(pred, axis=0, keep_dims=True, name='batchAvgPred')
        for iOut in range(len(names)):
            outName = names[iOut]
            lossDict['sqrLossesPerVar'+'/'+outName] = tf.reduce_mean(sqrLosses[:,iOut,:], axis=0, name='sqrLossesPerVar'+'/'+outName)
            lossDict['logSqrLosPerVar'+'/'+outName] = tf.identity(tf.log(lossDict['sqrLossesPerVar'+'/'+outName]) / tf.log(10.), name='logSqrLosPerVar'+'/'+outName)
            lossDict['absLossesPerVar'+'/'+outName] = tf.reduce_mean(absLosses[:,iOut,:], axis=0, name='absLossesPerVar'+'/'+outName)
            lossDict['logLossesPerVar'+'/'+outName] = tf.reduce_mean(loglosses[:,iOut,:], axis=0, name='logLossesPerVar'+'/'+outName)
            lossDict['meanYPerVar'+'/'+outName]     = tf.reduce_mean(batchAvgY[:,iOut,:], axis=0, name='meanYPerVar'+'/'+outName)
            lossDict['meanPredPerVar'+'/'+outName]  = tf.reduce_mean(batchAvgPred[:,iOut,:], axis=0, name='meanPredPerVar'+'/'+outName)
            lossDict['meanErrPerVar'+'/'+outName]   = tf.reduce_mean(tf.square(y[:,iOut,:] - batchAvgY[:,iOut,:]), axis=0, name='meanErrPerVar'+'/'+outName)
            lossDict['R2PerVar'+'/'+outName]        = tf.identity(1. - tf.divide(lossDict['sqrLossesPerVar'+'/'+outName] ,lossDict['meanErrPerVar'+'/'+outName]+1e-15), name='R2PerVar'+'/'+outName)
    with tf.name_scope('lossAvgLevel'):
        keys = list(lossDict.keys())
        for n in keys:
            lossDict[n.replace('PerVar', 'AvgLev')] = tf.reduce_mean(lossDict[n], axis=-1, name=n.replace('PerVar', 'AvgLev'))

    with tf.name_scope('loss'):
        lossDict['RMSE'] = tf.sqrt(tf.reduce_mean(sqrLosses), name='RMSE')
        lossDict['logRMSE'] = tf.identity(tf.log(lossDict['RMSE']) / tf.log(10.), name='logRMSE')
        lossDict['mse'] = tf.reduce_mean(sqrLosses, name='mse')
        lossDict['logloss'] = tf.reduce_mean(loglosses, name='logloss')
        lossDict['absloss'] = tf.reduce_mean(absLosses, name='absloss')
        lossDict['R2'] = tf.identity(1.- tf.divide(tf.reduce_sum(sqrLosses), tf.reduce_sum(tf.square(y - batchAvgY))), name='R2')
        
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