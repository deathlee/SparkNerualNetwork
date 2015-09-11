__author__ = 'zhuangli'
import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="/Users/zhuangli/Downloads/spark-1.4.1"

# Append pyspark  to Python Path
sys.path.append("/Users/zhuangli/Downloads/spark-1.4.1/python/")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.util import MLUtils
from pyspark.mllib.util import SparseVector
from MultiLabeledPoint import MultiLabeledPoint
import os
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

import random
logFile = "/Users/zhuangli/Downloads/spark-1.4.1/README.md"  # Should be some file on your system
sc = SparkContext("local", "Simple App")
def loadData(path):
    return MLUtils.loadLibSVMFile(sc, path)
def appendBias(vec):
    if isinstance(vec, SparseVector):
        newIndices = np.append(vec.indices+1, len(vec)+1)
        newValues = np.append(vec.values, 1.0)
        return SparseVector(len(vec) + 2, newIndices, newValues)
def aggregateData():
    multilabel=[]
    comments=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/comment_test")
    forwards=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/forward_test")
    likes=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/like_test")
    tests=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/test_test")
    comments=comments.collect()
    forwards=forwards.collect()
    likes=likes.collect()
    for i in range(len(comments)):
        features=appendBias(comments[i].features)
        multilabel.append(MultiLabeledPoint(comment=comments[i].label,forward=forwards[i].label,like=likes[i].label,features=features))
    return multilabel
multiLabeledPoint=sc.parallelize(aggregateData(),4)
#print len(multiLabeledPoint.collect()[2].features.toArray())
#print multiLabeledPoint.collect()[2].features
#print multiLabeledPoint.collect()[2].features.toArray()
a=multiLabeledPoint.collect()[2].features+multiLabeledPoint.collect()[3].features
print a
"""
global seed
"""
def add_random_key(it):
    seed = int(os.urandom(4).encode('hex'), 16)
    rs = np.random.RandomState(seed)
    return ((rs.rand(), x) for x in it)
def CountFromPartion(index,list_of_lists):
    final_iterator = []
    count=0
    for l in list_of_lists:
        count+=l.comment+l.forward+l.like
    final_iterator.append((index,count))
    return iter(final_iterator)
def partitionRDD(originlRDD,numPartitions):
    rdd_with_keys = (originlRDD.mapPartitions(add_random_key, preservesPartitioning=True))
    rdd=(rdd_with_keys.partitionBy(numPartitions).mapPartitions(sorted, preservesPartitioning=True).values())
    return rdd
def forwardPro():
    countRDD=multiLabeledPoint.mapPartitionsWithIndex(CountFromPartion)
    count=countRDD.collect()
