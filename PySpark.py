__author__ = 'zhuangli'
#import lib
import os
import sys
from SparkNerualNetwork import NerualNetwork
# Path for spark source folder
os.environ['SPARK_HOME']="/Users/zhuangli/PycharmProjects/Weibo/spark-1.4.1"
# Append pyspark  to Python Path
sys.path.append("/Users/zhuangli/PycharmProjects/Weibo/spark-1.4.1/python/")
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
import random
# setting spark environment
conf=SparkConf()
sc = SparkContext("local[1]", "Simple App")
#actition function and derivative function

# initial functions
def loadData(path):
    return MLUtils.loadLibSVMFile(sc, path)
def appendBias(vec):
    if isinstance(vec, SparseVector):
        newIndices = np.append(vec.indices+1, len(vec)+1)
        newValues = np.append(vec.values, 1.0)
        return SparseVector(len(vec) + 2, newIndices, newValues)
def appendTestBias(vec,dim):
    if isinstance(vec, SparseVector):
        newIndices = np.append(vec.indices+1, dim-1)
        newValues = np.append(vec.values, 1.0)
        return SparseVector(dim, newIndices, newValues)
def loadTestData(dim):
    test=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/test_test")
    test=test.collect()
    labelpoints=[]
    for point in test:
        features=appendTestBias(point.features,dim)
        labelpoints.append(LabeledPoint(label=0,features=features))
    return labelpoints
def aggregateTrainingData():
    multilabel=[]
    comments=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/comment_test")
    forwards=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/forward_test")
    likes=loadData("/Users/zhuangli/PycharmProjects/Weibo/data/like_test")
    comments=comments.collect()
    forwards=forwards.collect()
    likes=likes.collect()
    for i in range(len(comments)):
        features=appendBias(comments[i].features)
        multilabel.append(MultiLabeledPoint(comment=comments[i].label,forward=forwards[i].label,like=likes[i].label,features=features,dim=len(features)))
    return multilabel,multilabel[0].dim
#multiLabeledPoint=sc.parallelize(aggregateTrainingData(),4)
#print len(multiLabeledPoint.collect()[2].features.toArray())
#print multiLabeledPoint.collect()[2].features
#print multiLabeledPoint.collect()[2].features.toArray()
# random partition
def add_random_key(it):
    seed = int(os.urandom(4).encode('hex'), 16)
    rs = np.random.RandomState(seed)
    return ((rs.rand(), x) for x in it)
def partitionRDD(originlRDD,numPartitions):
    rdd_with_keys = (originlRDD.mapPartitions(add_random_key, preservesPartitioning=True))
    rdd=(rdd_with_keys.partitionBy(numPartitions).mapPartitions(sorted, preservesPartitioning=True).values())
    return rdd
def filterOut2FromPartion(index,list_of_lists):
    count=0
    for sub_list in list_of_lists:
        count+=sub_list.comment
    return iter(index,count)

points,dim=aggregateTrainingData()
multiLabeledPoint=sc.parallelize(points,1).cache()
testData=sc.parallelize(loadTestData(dim)).cache()
#filtered_lists = multiLabeledPoint.mapPartitionsWithIndex(filterOut2FromPartion)
#print(filtered_lists.collect())
nerualNetwork= NerualNetwork(layers=[dim,10,1])
nerualNetwork.train(multiLabeledPoint)
nerualNetwork.predict(testData)
