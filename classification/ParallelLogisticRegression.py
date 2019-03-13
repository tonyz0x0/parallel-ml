# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from LogisticRegression import readBeta, writeBeta, gradLogisticLoss, logisticLoss, lineSearch
from operator import add
from pyspark import SparkContext

"""
Parameters example: 
  --testdata "mushrooms/mushrooms.test" --beta "beta" --lam 0.0 --max_iter 20 --eps 0.1 --N 8 mushrooms/mushrooms.train
  --testdata "newsgroups/news.test" --beta "beta" --lam 0.0 --max_iter 20 --eps 0.1 --N 8 newsgroups/news.train
"""


def readDataRDD(input_file, spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The return value is an RDD containing tuples of the form
                 (SparseVector(x),y)             

    """
    return spark_context.textFile(input_file) \
        .map(eval) \
        .map(lambda (x, y): (SparseVector(x), y))


def getAllFeaturesRDD(dataRDD):
    """ Get all the features present in grouped dataset dataRDD.
 
	The input is:
            - dataRDD containing pairs of the form (SparseVector(x),y).  

        The return value is an RDD containing the union of all unique features present in sparse vectors inside dataRDD.
    """
    return dataRDD.keys()\
        .reduce(lambda x, y: x + y)\
        .keys()


def totalLossRDD(dataRDD, beta, lam=0.0):
    """  Given a sparse vector beta and a dataset  compute the regularized total logistic loss :

               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2

         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ
    """
    return dataRDD.map(lambda (x, y): logisticLoss(beta, x, y))\
        .reduce(lambda x, y: x + y)\
        + lam * beta.dot(beta)


def gradTotalLossRDD(dataRDD, beta, lam=0.0):
    """  Given a sparse vector beta and a dataset perform compute the gradient of regularized total logistic loss :

              ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β

         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ
    """
    return dataRDD.map(lambda (x, y): gradLogisticLoss(beta, x, y))\
        .reduce(lambda x, y: x + y)\
        + 2. * lam * beta


def test(dataRDD, beta):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.

        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0

                 TP = datapoints in (x,y) in P for which y=+1
                 FP = datapoints in (x,y) in P for which y=-1
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector β over data are defined as:

                 ACC(β,data) = ( #TP+#TN ) / (#P + #N)
                 PRE(β,data) = #TP / (#TP + #FP)
                 REC(β,data) = #TP/ (#TP + #FN)

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return values are
             - ACC, PRE, REC

    """
    _rdd = dataRDD.map(lambda (x, y): (beta.dot(x), y))
    P_RDD = _rdd.filter(lambda (x, y): x > 0)
    N_RDD = _rdd.filter(lambda (x, y): x <= 0)
    P = P_RDD.count()
    N = N_RDD.count()
    TP = P_RDD.filter(lambda (x, y): y == 1).count()
    FP = P_RDD.filter(lambda (x, y): y == -1).count()
    TN = N_RDD.filter(lambda (x, y): y == -1).count()
    FN = N_RDD.filter(lambda (x, y): y == 1).count()
    ACC = 1. * (TP + TN) / (P + N)
    PRE = 1. * TP / (TP + FP)
    REC = 1. * TP / (TP + FN)
    return ACC, PRE, REC

def train(dataRDD, beta_0, lam, max_iter, eps, test_data=None):
    k = 0
    gradNorm = 2 * eps
    beta = beta_0
    start = time()
    while k < max_iter and gradNorm > eps:
        obj = totalLossRDD(dataRDD, beta, lam)

        grad = gradTotalLossRDD(dataRDD, beta, lam)
        gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)

        fun = lambda x: totalLossRDD(dataRDD, x, lam)
        gamma = lineSearch(fun, beta, grad, obj, gradNormSq)

        beta = beta - gamma * grad
        if test_data == None:
            print 'k = ', k, '\tt = ', time() - start, '\tL(β_k) = ', obj, '\t||∇L(β_k)||_2 = ', gradNorm, '\tgamma = ', gamma
        else:
            acc, pre, rec = test(test_data, beta)
            print 'k = ', k, '\tt = ', time() - start, '\tL(β_k) = ', obj, '\t||∇L(β_k)||_2 = ', gradNorm, '\tgamma = ', gamma, '\tACC = ', acc, '\tPRE = ', pre, '\tREC = ', rec
        k = k + 1

    return beta, gradNorm, k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Logistic Regression.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('traindata', default=None,
                        help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata', default=None,
                        help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta',
                        help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float, default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--N', type=int, default=2, help='Level of parallelism')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Logistic Regression')

    if not args.verbose:
        sc.setLogLevel("ERROR")

    print 'Reading training data from', args.traindata
    traindata = readDataRDD(args.traindata, sc)
    traindata = traindata.repartition(args.N).cache()
    print 'Read', traindata.count(), 'data points with', len(getAllFeaturesRDD(traindata)), 'features in total'

    if args.testdata is not None:
        print 'Reading test data from', args.testdata
        testdata = readDataRDD(args.testdata, sc)
        testdata = testdata.repartition(args.N).cache()
        print 'Read', testdata.count(), 'data points with', len(getAllFeaturesRDD(testdata)), 'features'
    else:
        testdata = None

    beta0 = SparseVector({})

    print 'Training on data from', args.traindata, 'with λ =', args.lam, ', ε =', args.eps, ', max iter = ', args.max_iter
    beta, gradNorm, k = train(traindata, beta_0=beta0, lam=args.lam, max_iter=args.max_iter, eps=args.eps,
                              test_data=testdata)
    print 'Algorithm ran for', k, 'iterations. Converged:', gradNorm < args.eps
    print 'Saving trained β in', args.beta
    writeBeta(args.beta, beta)
