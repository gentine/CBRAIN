from __future__ import print_function

import os, time
from io import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

try:
	from beholder.beholder import Beholder
	from yellowfin import YFOptimizer
except:
	pass

from models import *
from climatLosses import *

def signLog(a, linearRegion=1):
    a /= linearRegion
    return tf.asinh(a/2)/np.log(10.0)
    return (tf.log(tf.nn.relu(a)+1) - tf.log(tf.nn.relu(-a)+1)) / np.log(10.0)

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        with tf.device("/cpu:0"):
            self.x, self.y = data_loader.get_inputs()
        print('self.x', self.x)
        print('self.y', self.y)

        self.step = tf.Variable(0, name='step', trainable=False)

        self.lr = tf.Variable(config.lr, name='lr', trainable=False)
        self.lr_update = tf.assign(self.lr, tf.maximum(self.lr * 0.5, config.lr_lower_boundary), name='lr_update')

        self.model_dir = config.model_dir
        print('self.model_dir: ', self.model_dir)

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        self.start_step = 0

        K.set_learning_phase(config.is_train)
        with tf.device("/gpu:0" if self.use_gpu else "/cpu:0"):
            if self.config.convo:
                self.build_model_convo()
            else:
                self.build_model()

            self.build_trainop()


        self.visuarrs = []
        y0 = self.y[:,:1]
        self.frameWorld = tf.log(1+tf.reshape(tf.reduce_mean(y0, axis=2), [self.data_loader.n_lat, -1]))
        with tf.name_scope('visuarrs'):
            maxWidth = self.data_loader.n_lon
            Xhb1c = tf.transpose(self.x[:maxWidth,:,:,0], [2,0,1])
            Yhb1c = tf.transpose(self.y[:maxWidth,:,:,0], [2,0,1])
            Phb1c = tf.transpose(self.pred[:maxWidth,:,:,0], [2,0,1])
            #Lhb1c = tf.transpose(self.losses['loss'][:maxWidth,:,:], [2,0,1])
            self.visuarrs += tf.unstack(Xhb1c, axis=-1)
            self.visuarrs += tf.unstack(Yhb1c, axis=-1)
            self.visuarrs += tf.unstack(Phb1c, axis=-1)
            #self.visuarrs += tf.unstack(Lhb1c, axis=-1)
            self.visuarrs += [tf.reshape(tf.reduce_mean(self.y[:,0], axis=1), [self.data_loader.n_lat, -1])]
            self.visuarrs += [tf.reshape(tf.reduce_mean(self.pred[:,0], axis=1), [self.data_loader.n_lat, -1])]
            self.visuarrs += [tf.reshape(tf.reduce_mean(self.y[:,1], axis=1), [self.data_loader.n_lat, -1])]
            self.visuarrs += [tf.reshape(tf.reduce_mean(self.pred[:,1], axis=1), [self.data_loader.n_lat, -1])]
            print("self.frameWorld", self.frameWorld.shape)
            for op in self.visuarrs:
                print('self.visuarrs', op)

        self.valStr = '' if config.is_train else '_val'
        self.saver = tf.train.Saver()# if self.config.is_train else None
        self.sumdir = self.model_dir + self.valStr
        self.summary_writer = tf.summary.FileWriter(self.sumdir)

        self.saveEverySec = 30
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=self.saveEverySec if self.config.is_train else 0,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        # start our custom queue runner's threads
        self.coord = tf.train.Coordinator()
        self.queueThreads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

    def train(self):
        try:
            visualizer = Beholder(session=self.sess, logdir='logs')
        except:
            pass
        totStep = 0
        for ep in range(1, self.config.epoch + 1):
            trainBar = trange(self.start_step, self.data_loader.NumBatch)
            #for i in range(70): self.sess.run(self.visuarrs)
            for step in trainBar:
                totStep += 1
                fetch_dict = {"optim": self.optim,
                        "visuarrs": self.visuarrs,
                        "frameWorld": self.frameWorld,
                        "x": self.x,
                        "y": self.y
                        }
                if step % self.config.log_step == 0:
                    fetch_dict.update({
                        "summary": self.summary_op,
                        "losses": self.losses,
                    })
                result = self.sess.run(fetch_dict)
                #print('x',np.mean(result['x'], axis=0))
                #print('y',np.mean(result['y'], axis=0))

                if step % self.config.log_step == 0:
                    self.summary_writer.add_summary(result['summary'], totStep)
                    self.summary_writer.flush()

                    losses = result['losses']
                    trainBar.set_description("epoch:{:03d}, L:{:.4f}, logL:{:+.3f}, RMSE:{:+.3f}, log10_RMSE:{:+.3f}, R2:{:+.3f}, q:{:d}, lr:{:.4g}". \
                        format(ep, losses['loss'], losses['logloss'], losses['RMSE'], np.log(losses['RMSE'])/np.log(10.), losses['R2'], 0, self.lr.eval(session=self.sess)))
                    for op in tf.global_variables():
                        npar = self.sess.run(op)
                        if 'Adam' not in op.name:
                            filename = self.model_dir+'/saveNet/'+op.name
                            try:
                                os.makedirs(os.path.dirname(filename))
                            except:
                                pass
                            np.save(filename, npar)

                visuarrs = result['visuarrs']#self.sess.run(self.visuarrs)
                frameWorld = result['frameWorld']#self.sess.run(self.visuarrs)
                try:
                    visualizer.update(arrays=visuarrs, frame=frameWorld)
                except:
                    pass#visualizer.update(arrays=visuarrs, frame=np.concatenate(visuarrs, axis=1))
                #for i in range(63+0*step//1000): self.sess.run(self.x)
                #if step % 100 == 0:
                #    self.sess.run(self.visuarrs)
                #time.sleep(0.1)
                if step % self.config.lr_update_step == self.config.lr_update_step - 1:
                        self.sess.run([self.lr_update])

            if ep % self.config.lr_update_epoch == self.config.lr_update_epoch - 1:
                    self.sess.run([self.lr_update])
        self.coord.request_stop()
        self.coord.join(self.queueThreads)

    def validate(self):
        numSteps = 50#self.data_loader.NumBatchValid
        trainBar = trange(self.start_step, numSteps)
        sleepTime = (self.saveEverySec/2) / numSteps
        print('sleepTime', sleepTime)
        for step in trainBar:
            fetch_dict = {} # does not train
            if True:#step % self.config.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "losses": self.losses,
                    "step": self.step
                })
            result = self.sess.run(fetch_dict)

            if True:#step % self.config.log_step == 0:
                self.summary_writer.add_summary(result['summary'], result['step'] + step)
                self.summary_writer.flush()

                losses = result['losses']
                trainBar.set_description("L:{:.6f}, logL:{:.6f}, RMSE:{:+.3f}, log10_RMSE:{:+.3f}, R2:{:+.3f}". \
                    format(losses['loss'], losses['logloss'], losses['RMSE'], np.log(losses['RMSE'])/np.log(10.),losses['R2']))
            time.sleep(sleepTime)
        self.coord.request_stop()
        self.coord.join(self.queueThreads)
        exit(0)

    def build_model(self):
        x = self.x
        print('x:', x)
        numChanOut = self.y.get_shape().as_list()[1]

        x = Flatten()(x)
        for nLay in self.config.hidden.split(','):
            nLay = int(nLay)
            print('x:', x)
            x = Dense(nLay, activation=self.config.act)(x)
        x = tf.expand_dims(tf.expand_dims(x, -1),-1)
        x = Conv2D(numChanOut, (1,1), padding='valid', data_format='channels_first')(x)
        print('self.pred:', x)
        self.pred = x#tf.reshape(x, self.y.get_shape())

    def build_model_convo(self):
        x = self.x
        print('x:', x)
        numChanOut = self.y.get_shape().as_list()[1]
        shapeX = self.x.get_shape().as_list()

        x = tf.reshape(x, [-1]+[shapeX[1]*shapeX[2]])
        #x = tf.layers.batch_normalization(x, axis=1, momentum=0.999, training=self.config.is_train)
        x = tf.reshape(x, shapeX)
        print('batchNorm(x):', x)
        for nLay in self.config.hidden.split(','):
            nLay = int(nLay)
            x = tf.pad(x, paddings=[[0,0],[0,0],[1,1],[0,0]], mode='SYMMETRIC')
            print('x:', x)
            if self.config.localConvo:
                x = LocallyConnected2D(nLay, (3,1), data_format='channels_first')(x)
            else:
                x = Conv2D(nLay, (3,1), padding='valid', data_format='channels_first')(x)
            x = LeakyReLU()(x)
        print('x:', x)
        x = Conv2D(numChanOut, (1,1), padding='valid', data_format='channels_first')(x)
#        x *= 1e-3
        print('self.pred:', x)
        self.pred = x#tf.reshape(x, self.y.get_shape())

    def build_trainop(self):
        y = self.y*1000
        p = self.pred
        print('y:', y)
        print('p:', p)
        numChanOut = y.get_shape().as_list()[1]
        print('numChanOut:', numChanOut)

        # Add ops to save and restore all the variables.
        self.losses = makeLossesPerLevel(y[:,:,:,0], p[:,:,:,0], self.data_loader.outputNames, self.config.lossfct)

        summaries = []
        for n,op in self.losses.items():
            if len(op.shape) < 1:
                summaries += [tf.summary.scalar(op.name, op)]
            else:
                summaries += [tf.summary.histogram(n, op)]
        summaries += [tf.summary.scalar("misc/lr", self.lr)]
        self.summary_op = tf.summary.merge(summaries)

        if self.config.is_train:
            if self.config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer
            elif self.config.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer
            elif self.config.optimizer == 'yf':
                optimizer = YFOptimizer
            else:
                raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

            optimizer = optimizer(self.lr)

            train_op = optimizer.minimize(self.losses['loss'], global_step=self.step)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print('update_ops', update_ops)
            with tf.control_dependencies(update_ops):
                self.optim = train_op

