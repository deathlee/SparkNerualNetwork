__author__ = 'zhuangli'
import numpy as np
import os
import sys
# Path for spark source folder
os.environ['SPARK_HOME']="/Users/zhuangli/Downloads/spark-1.4.1"
# Append pyspark  to Python Path
sys.path.append("/Users/zhuangli/Downloads/spark-1.4.1/python/")
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.util import SparseVector
from MultiLabeledPoint import MultiLabeledPoint
b=np.random.random((1, 4))
a=SparseVector(4,[1,3],[1,3])
print a.toArray().shape
print a.toArray().reshape(a.toArray().shape[0],1).dot(b)
print a.toArray().reshape(a.toArray().shape[0],1)