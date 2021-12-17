###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-136-117.ec2.internal
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

##############
# Parameters #
##############
MONTH = '202109*'

######################################################################
# Read Source Path for travel (brq or timeseries? Daily or monthly?) #
######################################################################
path = 's3a://ada-prod-data/etl/data/brq/agg/agg_brq/timeseries/daily/KH/'+MONTH+'/*.parquet'
ts = spark.read.parquet(path).select('ifa',explode('gps'))\
        .select('ifa','col.city','col.state','col.country','col.last_seen','col.brq_count')
ts = ts.filter(col('country') == 'KH').cache()
ts.printSchema()
ts.show(5,0)

# sort by ifa and date
cols = ['ifa','last_seen']
ts = ts.orderBy(*cols, ascending=True).cache()
ts.show(5,0)

#ts_.select('state').distinct().show(100,0)
#ts.select('country').distinct().show(100,0)


############################################
# Map to dual sim ifas only (Aug and Sept) #
############################################
path = 's3a://smart-bucket/rfm/'+MONTH+'/*.csv'
ds = spark.read.csv(path, header=True).cache()
ds.printSchema()
ds.show(5,0)

# Inner join
data = ts.join(ds,'ifa','inner').select('ifa','city','state','last_seen','brq_count').cache()
data.show(5,0)
#data.select('state').distinct().show(100,0)


# Add columns for date and time (split last_seen)
data2 = data.withColumn('date', split(col('last_seen'), ' ').getItem(0))
data2 = data2.withColumn('time', split(col('last_seen'), ' ').getItem(1))
data2 = data2.withColumn('day', date_format(col('last_seen'), 'EEEE'))
data2 = data2.withColumn('hour', F.hour('time'))

data2.show(5,0)
data2.printSchema()
#data2.select('hour').distinct().show(100,0)

# add column next_state and logic to take only same ifas
window = Window.partitionBy('ifa').orderBy(F.col('last_seen').asc())
df_data = data2.withColumn('next_ifa', lead(col('ifa')).over(window)).cache() # Boundary of 2 different ifas is marked by null! YAssss
df_data2 = df_data.withColumn('next_state', F.when(col('next_ifa') != 'null', lead(col('state')).over(window)).otherwise('null')).cache()

df_data.show(5,0)
df_data2.show(5,0)

# add column travel
df_data3 = df_data2.withColumn('travel', F.concat(col('state'), lit('_'), col('next_state'))).cache()
df_data3.show(5,0)

# Inter state Travelers
traveler = df_data3.filter( (col('state') != col('next_state')) & (~col('travel').like('%null%')) ).cache()
traveler.show(10,0)

# Groupby travel and count ifas
#travel = traveler.groupBy('travel').agg(count('ifa').alias('count')).sort(col('count'), ascending=False)
#travel.show(10,0) # These are instances of travel not unique ifas

'''
+-----------------------------+-----+
|travel                       |count|
+-----------------------------+-----+
|Kampong Thom_Phnom Penh      |84459|
|Phnom Penh_Kampong Thom      |75973|
|Phnom Penh_Kandal            |58391|
|Kandal_Phnom Penh            |58378|
|Banteay Meanchey_Phnom Penh  |25037|
|Phnom Penh_Banteay Meanchey  |24798|
|Kampong Speu_Phnom Penh      |18564|
+-----------------------------------+
'''

##
weekend = ['Saturday','Sunday']
morning = ['6','7','8','9','10','11','12']
afternoon =['13','14','15','16','17']
evening = ['18','19','20','21','22']
night = ['23','0','1','2','3','4','5']

traveler2 = traveler.withColumn('weekday_weekend', F.when(col('day').isin(weekend), F.lit('weekend')).otherwise('weekday'))
traveler2 = traveler2.withColumn('time_of_day', F.when(col('hour').isin(morning), F.lit('morning'))\
                .when(col('hour').isin(afternoon), F.lit('afternoon'))\
                .when(col('hour').isin(evening), F.lit('evening'))\
                .when(col('hour').isin(night), F.lit('night'))\
                ).cache()

traveler2.show(5,0)

# groupby ifa and travel count state (ifa level df)
final = traveler2.groupBy(['ifa','travel','weekday_weekend','time_of_day']).agg(count('state').alias('count')).sort(col('count'), ascending=False)
final.show(10,0)
'''
+------------------------------------+-----------------------+---------------+-----------+-----+
|ifa                                 |travel                 |weekday_weekend|time_of_day|count|
+------------------------------------+-----------------------+---------------+-----------+-----+
|747b809a-c866-4920-8a17-ff8424da0edb|Phnom Penh_Kandal      |weekday        |morning    |32   |
|747b809a-c866-4920-8a17-ff8424da0edb|Kandal_Phnom Penh      |weekday        |morning    |26   |
|fb820d81-03da-4d49-8bbe-9056d954e428|Kandal_Phnom Penh      |weekday        |afternoon  |24   |
|fb820d81-03da-4d49-8bbe-9056d954e428|Kampong Speu_Kandal    |weekday        |afternoon  |23   |
|02f7a526-9ebc-4094-b124-3a02b6f09097|Phnom Penh_Kandal      |weekday        |evening    |22   |
|3d53cd45-a6b1-4942-9aa4-389cdbd12144|Prey Veng_Kampong Cham |weekday        |morning    |22   |
|3d53cd45-a6b1-4942-9aa4-389cdbd12144|Kampong Cham_Prey Veng |weekday        |afternoon  |21   |
|fb820d81-03da-4d49-8bbe-9056d954e428|Phnom Penh_Kandal      |weekday        |morning    |20   |
|02f7a526-9ebc-4094-b124-3a02b6f09097|Kandal_Phnom Penh      |weekday        |morning    |20   |
|8d20dd26-aed7-4cbe-bbf6-ee99154bc13a|Kampong Thom_Phnom Penh|weekday        |evening    |20   |
+------------------------------------+-----------------------+---------------+-----------+-----+
'''

temp = traveler2.groupBy(['travel','weekday_weekend','time_of_day']).agg(count('state').alias('count')).sort(col('count'), ascending=False)
temp.show(10,0)


# Combine to and fro travels and sum count
#window1 = Window.partitionBy('ifa').orderBy(F.col('travel').asc())
#final_aug2 = final_aug.withColumn('next_travel', lead(col('travel')).over(window1)).cache()
#final_aug2.show(20,0)

# Define rfm transform segments
pp = spark.read.parquet('s3a://smart-bucket/rfm_transform/202108_202109/PP/*.parquet')
ss = spark.read.parquet('s3a://smart-bucket/rfm_transform/202108_202109/SS/*.parquet')
sp = spark.read.parquet('s3a://smart-bucket/rfm_transform/202108_202109/SP/*.parquet')
ps = spark.read.parquet('s3a://smart-bucket/rfm_transform/202108_202109/PS/*.parquet')
others = spark.read.parquet('s3a://smart-bucket/rfm_transform/202108_202109/Others/*.parquet')
all = spark.read.parquet('s3a://smart-bucket/rfm_transform/202108_202109/*/*.parquet')

# Join final_aug with rfm segments
#pp_aug = final.join(pp,'ifa','inner').cache()
#ss_aug = final.join(ss,'ifa','inner').cache()
#sp_aug = final.join(sp,'ifa','inner').cache()
#ps_aug = final.join(ps,'ifa','inner').cache()
#others_aug = final.join(others,'ifa','inner').cache()
#all_aug = final.join(all,'ifa','inner').cache()
#pp_aug.show(5,0)
#ss_aug.show(5,0)
#sp_aug.show(5,0)
#ps_aug.show(5,0)
#all_aug.show(5,0)

'''
+------------------------------------+--------------------+-----+
|ifa                                 |travel              |count|
+------------------------------------+--------------------+-----+
|0235347e-cf60-40fe-a7be-85d5b4465905|Phnom Penh_Kandal   |1    |
|0235347e-cf60-40fe-a7be-85d5b4465905|Kandal_Phnom Penh   |1    |
|083b0925-2fdb-466f-9d12-25b196f160d6|Phnom Penh_Prey Veng|1    |
|083b0925-2fdb-466f-9d12-25b196f160d6|Prey Veng_Phnom Penh|1    |
|083b0925-2fdb-466f-9d12-25b196f160d6|Phnom Penh_Kandal   |1    |
+------------------------------------+--------------------+-----+
'''

# Groupby travel and agg sum count
#pp_aug_df = pp_aug.groupBy('travel').agg(sum('count').alias('count')).sort(col('count'), ascending=False)
#pp_aug_df.show(20,0)
#ss_aug_df = ss_aug.groupBy('travel').agg(sum('count').alias('count')).sort(col('count'), ascending=False)
#ss_aug_df.show(20,0)
#sp_aug_df = sp_aug.groupBy('travel').agg(sum('count').alias('count')).sort(col('count'), ascending=False)
#sp_aug_df.show(20,0)
#ps_aug_df = ps_aug.groupBy('travel').agg(sum('count').alias('count')).sort(col('count'), ascending=False)
#ps_aug_df.show(20,0)

'''
+---------------------------+-----+
|travel                     |count|
+---------------------------+-----+
|Kampong Thom_Phnom Penh    |10431|
|Phnom Penh_Kampong Thom    |9716 |
|Kandal_Phnom Penh          |9582 |
|Phnom Penh_Kandal          |9565 |
|Banteay Meanchey_Phnom Penh|2317 |
|Phnom Penh_Banteay Meanchey|2248 |
|Kampong Speu_Phnom Penh    |2135 |
|Phnom Penh_Kampong Speu    |2108 |
|Phnom Penh_Takeo           |1512 |
|Takeo_Phnom Penh           |1488 |
|Siem Reap_Phnom Penh       |1309 |
|Phnom Penh_Siem Reap       |1272 |
|Battambang_Phnom Penh      |1257 |
|Phnom Penh_Battambang      |1197 |
|Stung Treng_Phnom Penh     |1058 |
|Phnom Penh_Stung Treng     |1051 |
|Phnom Penh_Preah Sihanouk  |1029 |
|Preah Sihanouk_Phnom Penh  |1018 |
|Kampot_Phnom Penh          |754  |
|Phnom Penh_Prey Veng       |739  |
+---------------------------+-----+
'''

######### Adding the time od day and day of week
# 4 times: Morning, afternoon, evening, night
# day of week: Monday to Sunday

#########################################################################################
# Cohort analysis - frequency of travel between the 4 top travel provinces (IFA level) #
##########################################################################################
# Travel between Kandal and Phnom Penh
list = ['Kandal_Phnom Penh','Phnom Penh_Kandal']
df = final.filter(col('travel').isin(list)).drop('travel')

df_weekday = df.groupBy('weekday_weekend').agg(sum('count').alias('count')).sort(col('count'), ascending=False)
df_weekday.show(5,0)
'''
+---------------+------+
|weekday_weekend|count |
+---------------+------+
|weekday        |118439|
|weekend        |41993 |
+---------------+------+
'''

df_tod = df.groupBy('time_of_day').agg(sum('count').alias('count')).sort(col('count'), ascending=False)
df_tod.show(5,0)
'''
+-----------+-----+
|time_of_day|count|
+-----------+-----+
|evening    |56823|
|morning    |47941|
|afternoon  |43408|
|night      |12260|
+-----------+-----+
'''
#
df1 = df.groupBy(['ifa']).agg(sum('count').alias('count')).sort(col('count'), ascending=False).cache()
df1.show(10,0)
df1.printSchema()

df1.selectExpr('percentile(count, 0.10)').show() # 1 # 1 | # 1 # 1
df1.selectExpr('percentile(count, 0.20)').show() # 2 # 2 | # 2 # 2
df1.selectExpr('percentile(count, 0.30)').show() # 2 # 2 | # 2 # 2
df1.selectExpr('percentile(count, 0.40)').show() # 2 # 2 | # 2 # 2
df1.selectExpr('percentile(count, 0.50)').show() # 2 # 2 | # 2 # 2
df1.selectExpr('percentile(count, 0.60)').show() # 2 # 4 | # 2 # 2
df1.selectExpr('percentile(count, 0.70)').show() # 2 # 4 | # 3 # 3
df1.selectExpr('percentile(count, 0.80)').show() # 4 # 6 | # 4 # 4
df1.selectExpr('percentile(count, 0.90)').show() # 6 # 12 | # 8 # 8

# Thresholds remain the same: 0-1, >1-4 , >4 (But the percentiles may change, so change accordingly)
rank = df1.filter(col('count').isNotNull()).agg(F.expr('percentile(count, array(0.10))')[0].alias('%10'),\
F.expr('percentile(count, array(0.80))')[0].alias('%80')) #,\ # First instance of 4 as the boundary
#F.expr('percentile(count, array(0.90))')[0].alias('%90'))

#tier_df2 = df1.withColumn("tier_id", F.lit(None))
#tier_df3 = tier_df2.withColumn('tier_id', when( (col('count') > 0) & (col('count') <= rank.collect()[0][0]) , 'TRV_01_001')\
#.when( (col('count') > rank.collect()[0][0]) & (col('count') <= rank.collect()[0][1]) , 'TRV_01_002')\
#.when( (col('count') > rank.collect()[0][1]) & (col('count') <= rank.collect()[0][2]) , 'TRV_01_003')\
#.when( (col('count') > rank.collect()[0][2]) , 'TRV_01_004'))

tier_df2 = df1.withColumn("tier_id", F.lit(None))
tier_df3 = tier_df2.withColumn('tier_id', when( (col('count') > 0) & (col('count') <= rank.collect()[0][0]) , 'TRV_01_001')\
.when( (col('count') > rank.collect()[0][0]) & (col('count') <= rank.collect()[0][1]) , 'TRV_01_002')\
.when( (col('count') > rank.collect()[0][1]) , 'TRV_01_003'))

tier_df = tier_df3.withColumn('tier_level', when( (col('tier_id') == 'TRV_01_001'), 'Low')\
.when( (col('tier_id') == 'TRV_01_002'), 'Mid')\
.when( (col('tier_id') == 'TRV_01_003'), 'High')).cache() #\
#.when( (col('tier_id') == 'TRV_01_004'), 'Ultra High')).cache()

tier_df.show(5,0)
tier_df.select('tier_level').distinct().show(100,0)

pp_joined = tier_df.join(pp,'ifa','inner')
ss_joined = tier_df.join(ss,'ifa','inner')
sp_joined = tier_df.join(sp,'ifa','inner')
ps_joined = tier_df.join(ps,'ifa','inner')
others_joined = tier_df.join(others,'ifa','inner')
all_joined = tier_df.join(all,'ifa','inner')

list = [pp_joined,ss_joined,ps_joined,sp_joined,others_joined,all_joined]
list2 = ['pp','ss','ps','sp','others','all']
for i,var in zip(list,list2):
    print(''+var+'')
    temp = i.groupBy('tier_level').agg(countDistinct('ifa').alias('ifa_count'))
    temp.show(5,0)

## Distribution of time of day
df2 = df.groupBy(['ifa','time_of_day']).agg(sum('count').alias('count')).sort(col('count'), ascending=False).cache()
df2.show(10,0)
df2.printSchema()

pp_joined = df2.join(pp,'ifa','inner')
ss_joined = df2.join(ss,'ifa','inner')
sp_joined = df2.join(sp,'ifa','inner')
ps_joined = df2.join(ps,'ifa','inner')
others_joined = df2.join(others,'ifa','inner')
all_joined = df2.join(all,'ifa','inner')

list = [pp_joined,ss_joined,ps_joined,sp_joined,others_joined,all_joined]
list2 = ['pp','ss','ps','sp','others','all']
for i,var in zip(list,list2):
    print(''+var+'')
    temp = i.groupBy('time_of_day').agg(sum('count').alias('count'))
    temp.show(5,0)


## Distribution of day of week
df3 = df.groupBy(['ifa','weekday_weekend']).agg(sum('count').alias('count')).sort(col('count'), ascending=False).cache()
df3.show(10,0)
df3.printSchema()

pp_joined = df3.join(pp,'ifa','inner')
ss_joined = df3.join(ss,'ifa','inner')
sp_joined = df3.join(sp,'ifa','inner')
ps_joined = df3.join(ps,'ifa','inner')
others_joined = df3.join(others,'ifa','inner')
all_joined = df3.join(all,'ifa','inner')

list = [pp_joined,ss_joined,ps_joined,sp_joined,others_joined,all_joined]
list2 = ['pp','ss','ps','sp','others','all']
for i,var in zip(list,list2):
    print(''+var+'')
    temp = i.groupBy('weekday_weekend').agg(sum('count').alias('count'))
    temp.show(5,0)




# Travel between Kandal and Phnom Penh
list = ['Kandal_Phnom Penh','Phnom Penh_Kandal']



# Travel between Banteay Meanchey and Phnom Penh
list = ['Banteay Meanchey_Phnom Penh','Phnom Penh_Banteay Meanchey']


# Travel between Kampong Thom and Phnom Penh
list = ['Kampong Thom_Phnom Penh','Phnom Penh_Kampong Thom'] # Change this to relevant travel journey


############
