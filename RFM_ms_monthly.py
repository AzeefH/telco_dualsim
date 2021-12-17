###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-128-20.ec2.internal
ssh -i ~/emr_key.pem $url

pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list --num-executors 25 --conf "spark.executor.memoryOverhead=2048" --executor-memory 9g --conf "spark.driver.memoryOverhead=6144" --driver-memory 50g --executor-cores 3 --driver-cores 5 --conf "spark.default.parallelism=150" --conf "spark.sql.shuffle.partitions=150" --conf "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2"

####
#pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
#pyspark --packages $pkg_list


from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.sql.types as T
import csv
import pandas as pd
import numpy as np
import sys
from pyspark.sql import Window
from pyspark.sql.functions import rank, col
#import geohash2 as geohash
#import pygeohash as pgh
from functools import reduce
from pyspark.sql import *
from pyspark import StorageLevel

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)





###########
