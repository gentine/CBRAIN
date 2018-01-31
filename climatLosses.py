from __future__ import print_function

import tensorflow as tf
import keras.backend as K
from keras.layers import ELU
from keras import losses

def makeLossesPerVar(y, pred, names, lossfct):
    print('makeLossesPerVar')
    print('y:', y)
    print('pred:', pred)
    tf.assert_equal(y.shape[1:], pred.shape[1:])
    numChanOut = y.get_shape().as_list()[1]
    numLevels = y.get_shape().as_list()[2]
    assert(numChanOut == len(names))
    print('numChanOut:', numChanOut)
    print(names)
    lossDict = {}   
    with tf.name_scope('metrics'):
        batchAvgY = tf.reduce_mean(y, axis=0, keep_dims=True, name='batchAvgY')
        batchAvgPred = tf.reduce_mean(pred, axis=0, keep_dims=True, name='batchAvgPred')
        emaVariable = tf.train.ExponentialMovingAverage(decay=0.9999)
        maintain_averages_op = emaVariable.apply([batchAvgY, batchAvgPred])
        with tf.control_dependencies([maintain_averages_op]):
            error = tf.identity(y - pred, name='error')
        emaY = emaVariable.average(batchAvgY)
        emaPred = emaVariable.average(batchAvgPred)
        rescaledY = y / (emaY + 1e-6)
        rescaledError = tf.identity(error + rescaledY - pred, name='rescaledError')
        sqrLosses = tf.square(error, name='sqrLosses')
        absLosses = tf.abs(error, name='absLosses')
        loglosses = tf.divide(tf.log(absLosses), tf.log(10.0), name='loglosses')

    with tf.name_scope('PerVar'):
        for iOut in range(len(names)):
            outName = names[iOut]
            lossDict['sqrLossesPerVar'+'/'+outName] = tf.reduce_mean(sqrLosses[:,iOut,:], axis=0, name='sqrLossesPerVar'+'/'+outName)
            lossDict['logSqrLosPerVar'+'/'+outName] = tf.identity(tf.log(lossDict['sqrLossesPerVar'+'/'+outName]) / tf.log(10.), name='logSqrLosPerVar'+'/'+outName)
            lossDict['absLossesPerVar'+'/'+outName] = tf.reduce_mean(absLosses[:,iOut,:], axis=0, name='absLossesPerVar'+'/'+outName)
            lossDict['logLossesPerVar'+'/'+outName] = tf.reduce_mean(loglosses[:,iOut,:], axis=0, name='logLossesPerVar'+'/'+outName)
            lossDict['meanYPerVar'+'/'+outName]     = tf.reduce_mean(emaY[:,iOut,:], axis=0, name='meanYPerVar'+'/'+outName)
            lossDict['meanPredPerVar'+'/'+outName]  = tf.reduce_mean(emaPred[:,iOut,:], axis=0, name='meanPredPerVar'+'/'+outName)
            lossDict['meanErrPerVar'+'/'+outName]   = tf.reduce_mean(tf.square(y[:,iOut,:] - emaY[:,iOut,:]), axis=0, name='meanErrPerVar'+'/'+outName)
            lossDict['R2PerVar'+'/'+outName]        = ELU(name='R2PerVar'+'/'+outName)(1. - tf.divide(lossDict['sqrLossesPerVar'+'/'+outName] ,lossDict['meanErrPerVar'+'/'+outName]))
    with tf.name_scope('lossAvgVar'):
        keys = list(lossDict.keys())
        for n in keys:
            lossDict[n.replace('PerVar', 'AvgVar')] = tf.reduce_mean(lossDict[n], axis=-1, name=n.replace('PerVar', 'AvgVar'))
    with tf.name_scope('PerLev'):
        for iLev in range(numLevels):
            outName = str(iLev)
            lossDict['sqrLossesPerLev'+'/'+outName] = tf.reduce_mean(sqrLosses[:,:,iLev], axis=0, name='sqrLossesPerLev'+'/'+outName)
            lossDict['logSqrLosPerLev'+'/'+outName] = tf.identity(tf.log(lossDict['sqrLossesPerLev'+'/'+outName]) / tf.log(10.), name='logSqrLosPerLev'+'/'+outName)
            lossDict['absLossesPerLev'+'/'+outName] = tf.reduce_mean(absLosses[:,:,iLev], axis=0, name='absLossesPerLev'+'/'+outName)
            lossDict['logLossesPerLev'+'/'+outName] = tf.reduce_mean(loglosses[:,:,iLev], axis=0, name='logLossesPerLev'+'/'+outName)
            lossDict['meanYPerLev'+'/'+outName]     = tf.reduce_mean(emaY[:,:,iLev], axis=0, name='meanYPerLev'+'/'+outName)
            lossDict['meanPredPerLev'+'/'+outName]  = tf.reduce_mean(emaPred[:,:,iLev], axis=0, name='meanPredPerLev'+'/'+outName)
            lossDict['meanErrPerLev'+'/'+outName]   = tf.reduce_mean(tf.square(y[:,:,iLev] - emaY[:,:,iLev]), axis=0, name='meanErrPerLev'+'/'+outName)
            lossDict['R2PerLev'+'/'+outName]        = ELU(name='R2PerLev'+'/'+outName)(1. - tf.divide(lossDict['sqrLossesPerLev'+'/'+outName] ,lossDict['meanErrPerLev'+'/'+outName]))
    with tf.name_scope('lossAvgLev'):
        keys = [k for k in list(lossDict.keys()) if 'Lev' in k]
        for n in keys:
            lossDict[n.replace('PerLev', 'AvgLev')] = tf.reduce_mean(lossDict[n], axis=-1, name=n.replace('PerLev', 'AvgLev'))

    with tf.name_scope('loss'):
        lossDict['RMSE'] = tf.sqrt(tf.reduce_mean(sqrLosses), name='RMSE')
        lossDict['logRMSE'] = tf.identity(tf.log(lossDict['RMSE']) / tf.log(10.), name='logRMSE')
        lossDict['mse'] = tf.reduce_mean(sqrLosses, name='mse')
        lossDict['logloss'] = tf.reduce_mean(loglosses, name='logloss')
        lossDict['absloss'] = tf.reduce_mean(absLosses, name='absloss')
        lossDict['R2'] = ELU(name='R2')(1.- tf.divide(tf.reduce_sum(sqrLosses), tf.reduce_sum(tf.square(y - batchAvgY))))
        lossDict['mape'] = tf.reduce_mean(losses.mean_absolute_percentage_error(y, pred), name="mape")
        
        # choose cost function
        lossDict['loss'] = lossDict[lossfct]
        if lossfct=="R2":
            lossDict['loss'] = -lossDict[lossfct]

        lossDict['loss'] = tf.identity(lossDict['loss'], name="loss")

    for n in lossDict.keys():
        print(n, lossDict[n])
    return lossDict
