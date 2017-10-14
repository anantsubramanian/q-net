import numpy as np
import tensorflow as tf
import sys
import os
import json

sys.path.append("./src/")
from utils.squad_utils import ObjDict


def XavierInit(fanIn, fanOut, shape, floatType):
    upperBound = np.sqrt(6 / float(fanIn + fanOut) )
    lowerBound = -np.sqrt(6 / float(fanIn + fanOut) )
    return tf.random_uniform(shape, minval=lowerBound, maxval=upperBound, dtype=floatType)


class TfRNNCell(object):
    '''
    generate recurrent neural network cells
    '''
    def __init__(self, nHidden, nLayer=1, keepProb=1.0, cellType="lstm"):
        self.nHidden = nHidden
        self.nLayer = nLayer
        self.keepProb = keepProb
        self.cellType = cellType


    def GetCell(self):
        if self.cellType == "rnn":
            cell = tf.nn.rnn_cell.BasicRNNCell(self.nHidden)
        elif self.cellType == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.nHidden)
        elif self.cellType == "gru":
            cell = tf.nn.rnn_cell.GRUCell(self.nHidden)
        else:
            raise Exception("Cell type " + self.cellType + " not supported!")

        if self.keepProb < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                output_keep_prob=self.keep_prob)
        if self.nLayer > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell( [cell] * self.nLayer)
        return cell


class TfTraining(object):
    '''
    training engine wrapper for tensorflow library
    '''
    def __init__(self, model, evaluator, lrFunc, optAlgo="SGD", 
        maxGradNorm=10, maxIter=0, FLAGS=None):
        '''
        @param model: a model class requiring the following:
        1. method loss which defines the loss op of the tensorflow computational graph.
        2. member trainVars. It is the tensor list from tf.trainable_variables().
        3. method GetNextTrainBatch: automatically traversing all the data. The output
        4. method PredictTrainSamples: give {qaid: answerStr} dict of the predictions
        on training samples
        5. method PredictEvalSamples: give {qaid: answerStr} dict of the prediction 
        on evaluation data (dev or test)
        of this function should be the feed_dict to tf.run
        @param lrFunc: customized stepSize rule generation. 
        The argument is a single iterNo.
        @param optAlgo: one of "SGD", "Adam", "momentume"
        @param FLAGS: tf FLAGS specifying at least the following
        flags.DEFINE_string("summaryPath", None, summaryPath)
        flags.DEFINE_string("ckptPath", None, ckptPath)
        flags.DEFINE_string("predPath", None, predPath)
        flags.DEFINE_integer("summaryFlushInterval", 100)
        flags.DEFINE_integer("evalInterval", 500)
        flags.DEFINE_integer("ckptInterval", 500)
        '''
        self.model = model
        self.session = None
        self.evaluator = evaluator
        self.lrFunc = lrFunc
        self.maxIter = maxIter
        self.lr = tf.Variable(0.0, trainable=False)
        self.FLAGS = FLAGS
        self.debug = []

        self.grads = tf.gradients(self.model.loss, self.model.trainVars)
        self.gradsClip, _ = tf.clip_by_global_norm(self.grads, maxGradNorm)
        
        if optAlgo == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif optAlgo == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr, beta2=0.95)
        elif optAlgo == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.975)
        elif optAlgo == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(self.lr)
        else:
            raise Exception("Optimizer " + optAlgo + " is not supported by TF training wrapper!")
        
        self.training = optimizer.apply_gradients(zip(self.gradsClip, self.model.trainVars) )        
        self.saver = tf.train.Saver()


    def SaveCheckPoint(self, sess, path):
        '''
        the evaluation results on evaluation and training data
        '''
        save_path = self.saver.save(sess, path + "/model.ckpt", global_step=self.nIter + 1)
        print("Check point model saved in " + path + " " + str(self.nIter) )


    def SaveEvaluation(self, predTrain, predEval, path):
        with open(path + "/train.pred-" + str(self.nIter), "w") as fp:
            json.dump(predTrain, fp, default=ObjDict)
        with open(path + "/eval.pred-" + str(self.nIter), "w") as fp:
            json.dump(predEval, fp, default=ObjDict)


    def Run(self, session, summarizer):
        self.session = session
        summary, summaryWriter = summarizer
        for i in range(self.maxIter):
            self.nIter = i
            inputDict = self.model.GetNextTrainBatch()
            inputDict[self.lr] = self.lrFunc(i)
            # print "start the new step!"
            results = self.session.run([self.training, self.model.loss, summary], feed_dict=inputDict)
            print str(i), " iter loss: ", str(results[1] )
            summaryWriter.add_summary(results[2], i)
            # flush summary
            if i % self.FLAGS.summaryFlushInterval == 0 or i == self.maxIter - 1:
                summaryWriter.flush()
            # save predictons
            if (i % self.FLAGS.evalInterval == 0 or i == self.maxIter - 1) and i != 0:
                trainPrediction = self.model.PredictTrainSamples(self.session)    
                trainSetRes = self.evaluator.EvaluatePrediction(self.model.trainSamples, trainPrediction)
                evalPrediction = self.model.PredictEvalSamples(self.session)
                evalSetRes = self.evaluator.EvaluatePrediction(self.model.evalSamples, evalPrediction)
                self.SaveEvaluation(trainPrediction, evalPrediction, self.FLAGS.predPath)
            # save model check point
            if i % self.FLAGS.ckptInterval == 0 or i == self.maxIter - 1:
                self.SaveCheckPoint(session, self.FLAGS.ckptPath)
