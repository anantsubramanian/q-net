import numpy as np
import tensorflow as tf
import os
import sys
import json


def MaxMarginLoss(paScores, naScores, naGroupPos, floatType, idType, batchSize):
    # calculate scores for positive and negative answers
    '''
    @param paScores: tensor in shape [nBatchSize, 1]
    @param naScores: tensor in shape [nTotalNaSampleForTheBatch, 1]
    @param naGroupPos : it indicates the starting index of the negative answers
    @param floatType, idType: need to be consistent with graph constructed by a model
    useful for compatability in some of the ops.
    corresponding to each sample. In the format [nSamples + 1].
    It must be guaranteed that every sample has at least 1 negative answer
    We output a list of assertions and evaluate it when session.run.
    '''
    naScoreList = []
    assertions = []
    for i in xrange(batchSize):
        # TODO better way to slice the tensor
        beginPos = tf.concat(0, [naGroupPos[i:(i+1) ], tf.constant([0,], dtype=idType) ] )
        nNaSample = naGroupPos[ (i+1):(i+2) ] - naGroupPos[i]
        # Assert the sample has non-zero negative answers
        assertions.append(tf.Assert(tf.greater(naGroupPos[i+1] - naGroupPos[i], 0), \
            ["Zero negative answers in max margin loss!"] ) )
        sliceSize = tf.concat(0, [nNaSample, tf.constant( [-1, ], dtype=idType) ] )
        # TODO the current version seems to require beginPos and sliceSize to be int32
        # however other places required int64. we keep the global setting to be int64
        # while the others 
        naScoreSlice = tf.slice(naScores, tf.cast(beginPos, tf.int32), tf.cast(sliceSize, tf.int32) )
        naScoreList.append(tf.reduce_max(naScoreSlice, keep_dims=True) )
    naScores = tf.concat(0, naScoreList)
    loss = tf.reduce_mean(tf.maximum(naScores - paScores + 1, 0), name="max-margin-loss")
    return loss, paScores, naScores, assertions


def NCELoss(paScores, naScores, naGroupPos, floatType, idType, batchSize):
    # calculate scores for positive and negative answers
    naScoreList = []
    for i in xrange(batchSize):
        # TODO better way to slice the tensor
        beginPos = tf.concat(0, [naGroupPos[i:(i+1) ], tf.constant([0,], dtype=idType) ] )
        nNaSample = naGroupPos[ (i+1):(i+2) ] - naGroupPos[i]
        sliceSize = tf.concat(0, [nNaSample, tf.constant( [-1, ], dtype=idType) ] )
        naScoreSlice = tf.slice(naScores, beginPos, sliceSize)
        if floatType == tf.float32:
            scores = tf.div(naScoreSlice, tf.to_float(nNaSample) )
        elif floatType == tf.float64:
            scores = tf.div(naScoreSlice, tf.to_double(nNaSample) )
        else:
            raise Exception("The float type is not supported!")
        naScoreList.append(scores)
    naScores = tf.concat(0, naScoreList)

    # use reduce_sum / batchSize in case there are samples
    # with zero negative answer
    loss = tf.reduce_sum(naScores) / tf.constant(batchSize, dtype=floatType) \
        - tf.reduce_mean(paScores)
    return loss, paScores, naScores 


def CrossEntLoss(paScores, naScores, naGroupPos, floatType, idType, batchSize):
    # calculate scores for positive and negative answers
    naScoreList = []    
    for i in xrange(batchSize):
        # TODO better way to slice the tensor
        beginPos = tf.concat(0, [naGroupPos[i:(i+1) ], tf.constant([0,], dtype=idType) ] )
        nNaSample = naGroupPos[ (i+1):(i+2) ] - naGroupPos[i]
        sliceSize = tf.concat(0, [nNaSample, tf.constant( [-1, ], dtype=idType) ] )
        naScoreSlice = tf.slice(naScores, beginPos, sliceSize)
        logits = tf.transpose(tf.concat(0, [paScores[i:(i+1), :], naScoreSlice] ) )
        if i == 0:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits, tf.zeros( [1], dtype=tf.int64) )
        else:
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits, tf.zeros( [1], dtype=tf.int64) )
    loss /= batchSize
    loss = tf.squeeze(loss)
    return loss, paScores, naScores 



