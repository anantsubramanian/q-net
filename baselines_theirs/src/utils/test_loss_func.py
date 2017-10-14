import tensorflow as tf
from loss_func import *

sys.path.append("./src/")
from learning_baseline.context_rnn import ContextRnnAgent
from learning_baseline.test_context_rnn import GetContextRnnAgent


def TestNCELoss():
    np.random.seed(0)
    seedList = np.random.randint(0, 10000000, [100, ] )
    
    agent = GetContextRnnAgent()
    for i in seedList:
        # the assertion may fail if using float32
        with tf.variable_scope("TestNCELoss_" + str(i) ):
            print "using seed ", i
            np.random.seed(i)
            floatType = tf.float64
            idType = tf.int32
            nQ = 10
            nCodeDim = 5
            nNa = np.random.randint(0, nQ * 3, nQ)
            qCodeNp = np.random.uniform(0, nQ * 10, [nQ, nCodeDim] )
            qCode = tf.constant(qCodeNp, floatType)
            paCodeNp = np.random.uniform(0, nQ * 10, [nQ, nCodeDim] )
            paCode = tf.constant(paCodeNp, floatType)
            naCodeNp = np.random.uniform(0, nQ * 10, [np.sum(nNa), nCodeDim] )
            naCode = tf.constant(naCodeNp, floatType)
            naGroupPosNp = np.hstack( (np.array( (0, ) ), np.cumsum(nNa) ) )
            naGroupPos = tf.constant(naGroupPosNp, dtype=idType)
            paScores, naScores, bias = agent.GetPaAndNaScores(qCode, paCode, naCode, naGroupPos, floatType, idType, nQ)
            loss, _, _ = NCELoss(paScores, naScores, naGroupPos, floatType, idType, nQ)
            # get tf based loss value
            with tf.Session() as session:
                with tf.device("/cpu:0"):
                    session.run(tf.initialize_all_variables() )
                    res = session.run( [loss, bias, paScores, naScores] )
            lossTf = res[0]
            scoreBias = res[1]
            # get numpy based value
            lossNp = 0
            for i in range(nQ):
                paScore = np.dot(qCodeNp[i, :], paCodeNp[i, :].T) + scoreBias
                lossNp -= paScore
                naScore = 0.0
                if naGroupPosNp[i + 1] - naGroupPosNp[i] != 0:
                    naScore = np.dot(qCodeNp[i, :], naCodeNp[naGroupPosNp[i]:naGroupPosNp[i + 1] ].T) + scoreBias
                    lossNp += np.mean(naScore)
            lossNp /= nQ

            print lossNp, lossTf
            assert abs(lossNp - lossTf) < 1e-8 * abs(lossNp)
    print "NCE loss function test passed!"


def TestMaxMarginLoss():
    np.random.seed(0)
    seedList = np.random.randint(0, 10000000, [100, ] )

    agent = GetContextRnnAgent()
    for i in seedList:
        # the assertion may fail if using float32
        with tf.variable_scope("TestMaxMarginLoss_" + str(i) ):
            print "using seed ", i
            # the assertion may fail if using float32
            floatType = tf.float64
            idType = tf.int32
            nQ = 10
            nCodeDim = 5
            nNa = np.random.randint(1, 3 * nQ, nQ)
            qCodeNp = np.random.uniform(0, nQ * 10, [nQ, nCodeDim] )
            qCode = tf.constant(qCodeNp, floatType)
            paCodeNp = np.random.uniform(0, nQ * 10, [nQ, nCodeDim] )
            paCode = tf.constant(paCodeNp, floatType)
            naCodeNp = np.random.uniform(0, nQ * 10, [np.sum(nNa), nCodeDim] )
            naCode = tf.constant(naCodeNp, floatType)
            naGroupPosNp = np.hstack( (np.array( (0, ) ), np.cumsum(nNa) ) )
            naGroupPos = tf.constant(naGroupPosNp, dtype=idType)
            
            paScores, naScores, bias = agent.GetPaAndNaScores(qCode, paCode, naCode, naGroupPos, floatType, idType, nQ)
            loss, _, _, assertions = MaxMarginLoss(paScores, naScores, naGroupPos, floatType, idType, nQ)
            # get tf based loss value
            with tf.Session() as session:
                with tf.device("/cpu:0"):
                    session.run(tf.initialize_all_variables() )
                    res = session.run( [loss, bias, naGroupPos] + assertions)

            lossTf = res[0]
            scoreBias = res[1]

            # get numpy based value
            lossNp = 0
            for i in range(nQ):
                paScore = np.dot(qCodeNp[i, :], paCodeNp[i, :].T) + scoreBias
                naScore = 0.0
                if naGroupPosNp[i + 1] - naGroupPosNp[i] != 0:
                    naScore = np.dot(qCodeNp[i, :], naCodeNp[naGroupPosNp[i]:naGroupPosNp[i + 1] ].T) + scoreBias
                    lossNp += max(np.max(naScore) - paScore + 1, 0)
            lossNp /= nQ

            print lossNp, lossTf
            assert abs(lossNp - lossTf) < 1e-8 * abs(lossNp)
    print "MaxMargin loss function test passed!"


def TestCrossEntLoss():
    np.random.seed(0)
    seedList = np.random.randint(0, 10000000, [100, ] )
    
    agent = GetContextRnnAgent()
    for i in seedList:
        # the assertion may fail if using float32
        with tf.variable_scope("TestCrossEntLoss_" + str(i) ):
            print "using seed ", i
            np.random.seed(i)
            floatType = tf.float64
            idType = tf.int32
            nQ = 10
            nCodeDim = 5
            nNa = np.random.randint(0, 3 * nQ, nQ)
            qCodeNp = np.random.uniform(0, 1, [nQ, nCodeDim] )
            qCode = tf.constant(qCodeNp, floatType)
            paCodeNp = np.random.uniform(0, 1, [nQ, nCodeDim] )
            paCode = tf.constant(paCodeNp, floatType)
            naCodeNp = np.random.uniform(0, 1, [np.sum(nNa), nCodeDim] )
            naCode = tf.constant(naCodeNp, floatType)
            naGroupPosNp = np.hstack( (np.array( (0, ) ), np.cumsum(nNa) ) )
            naGroupPos = tf.constant(naGroupPosNp, dtype=idType)
            paScores, naScores, bias = agent.GetPaAndNaScores(qCode, paCode, naCode, naGroupPos, floatType, idType, nQ)
            loss, _, _ = CrossEntLoss(paScores, naScores, naGroupPos, floatType, idType, nQ)
            # get tf based loss value
            with tf.Session() as session:
                with tf.device("/cpu:0"):
                    session.run(tf.initialize_all_variables() )
                    res = session.run( [loss, bias, paScores, naScores] )
            lossTf = res[0]
            scoreBias = res[1]
            # get numpy based value
            lossNp = 0
            for i in range(nQ):
                paScore = np.dot(qCodeNp[i, :], paCodeNp[i, :].T) + scoreBias
                score = paScore
                if naGroupPosNp[i + 1] - naGroupPosNp[i] != 0:
                    naScore = np.dot(qCodeNp[i, :], naCodeNp[naGroupPosNp[i]:naGroupPosNp[i + 1] ].T) + scoreBias
                    score = np.hstack( (score, naScore) )
                logits = np.exp(score - np.max(score) )
                lossNp += -np.log(logits[0] / np.sum(logits) )
            lossNp /= nQ

            print lossNp, lossTf
            assert abs(lossNp - lossTf) < 1e-8 * abs(lossNp)
    print "CrossEnt loss function test passed!"



if __name__ == "__main__":
    '''
    We define a scope for each test, otherwise there will be variable naming conflict.
    '''
    # with tf.variable_scope("NCE_loss_test"):
    #     TestNCELoss()
    # with tf.variable_scope("max_margin_loss_test"):
    #     TestMaxMarginLoss()

    with tf.variable_scope("cross_entropy_loss_test"):
        TestCrossEntLoss()
