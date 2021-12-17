###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-130-94.ec2.internal
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
# Paramaters #
##############
COUNTRY ='KH'
MONTH = '202111'


#####################
# Get dual sim IFAs #
#####################
data_path = "s3a://smart-bucket/ts_carrier_dataprep/202111/*.parquet"
ts_df = spark.read.parquet(data_path).cache()
ts_df.show(5,0)

# For monthly (USE THIS FOR MONTHLY)
ts_df = ts_df.where(col("carrier").isNotNull()).where(col("network")=="cellular").withColumn('month', ts_df.date.substr(1, 6)).cache()
ts_df.show(5,0)

ms = ts_df.select("ifa","carrier").groupBy("ifa").agg(collect_set("carrier").alias("carriers"),countDistinct("carrier").alias("total_carriers"))
ms = ms.where(col("total_carriers")>1).select("ifa").cache()

ts_nfreq = ts_df.groupBy("ifa","carrier").agg(countDistinct("date").alias("ndays"),countDistinct("month").alias("nmonths"),sum("brq_count").alias("sum(brq_count)"))

ts_freq = ts_df.groupBy("ifa","carrier","month","date").agg(sum("brq_count").alias("brq_count"))
join = ms.join(ts_freq,on=['ifa'],how='inner')

df = join.withColumn('last_date', lit(join.agg(F.max(join.date)).first()[0])).withColumn('last_month', lit(join.agg(F.max(join.month)).first()[0]))
df = df.withColumn('period_months',(col("last_month")-col("month")+1).cast("integer")).withColumn('day_since',(col("last_date")-col("date")+1).cast("integer"))
df.show(5,0)

#nmonths = total of months
#ndays = total of days (for nmonths data)
#sum(brq_count) = total of brq_count (for nmonths data)
#brq_count = total of brq_count (for one daily)
#period_months = last month - month
#day_since = last date - date


#########################
#   Ndays (Frequency)   #
#########################
udf = ts_nfreq.select("ifa","carrier","ndays")
# ndays = sum_ndays


########################
# Total BRQ (Monetary) #
########################
udf2 = df.drop("last_date").drop("last_month")
#sum_brq = brq_count


########################
#         RFM          #
########################

rdf = udf2.groupBy('ifa','carrier').agg(F.min('date').alias('earliest_date'),F.max('date').alias('latest_date'),F.max('day_since').alias('first_seen'),F.min('day_since').alias('last_seen'),F.min('period_months').alias('min_period'),F.max('period_months').alias('max_period'),F.sum('brq_count').alias('brq_count'))
rdf.show(5,0)

rdf2 = rdf.join(udf, on=['ifa','carrier']).withColumnRenamed('ndays', 'data_usage_freq')

## R,F,M variables
rfm = rdf2.withColumn('usage_frequency',F.col('data_usage_freq')/1).withColumn('brq_m',F.col('brq_count')/1).select('ifa','carrier','last_seen','usage_frequency','brq_m','data_usage_freq','brq_count').cache()
#nmonths = rdf.agg(F.max(rdf.max_period)).first()[0])
#rfm = rdf2.withColumn('usage_frequency',F.col('data_usage_freq')/nmonths).withColumn('brq_m',F.col('brq_count')/nmonths).select('ifa','carrier','last_seen','usage_frequency','brq_m','data_usage_freq','brq_count').cache()

rfm.show(4,0)


#########################
#   Distribution Test   #
#########################
#r_dist = rfm.select('ifa', 'last_seen').withColumn('last_seen_weeks', (F.col('last_seen')/7)).distinct()
#r_dist = r_dist.groupBy('last_seen_weeks').agg(F.count('ifa').alias('ifa_count')).sort(col('last_seen_weeks'), ascending = True)
r_dist = rfm.groupBy('last_seen','carrier').agg(F.countDistinct('ifa').alias('ifa_count')).sort(col('last_seen'), ascending = True).cache()
r_dist.show(20,0)

#r_dist.coalesce(1).write.csv('s3a://ada-dev/azeef/projects/'+COUNTRY+'/202110/smart_dualsim/'+MONTH+'/rfm/dist_test/r_dist', header = True, mode='overwrite')

### F Distribution
f_dist = rfm.groupBy('data_usage_freq','carrier').agg(F.countDistinct('ifa').alias('ifa_count')).sort(col('data_usage_freq'), ascending = True).cache()
f_dist.show(20,0)

#f_dist.coalesce(1).write.csv('s3a://ada-dev/azeef/projects/'+COUNTRY+'/202110/smart_dualsim/'+MONTH+'/rfm/dist_test/f_dist', header = True, mode='overwrite')

### M Distribution
m_dist = rfm.groupBy('brq_m','carrier').agg(F.countDistinct('ifa').alias('ifa_count')).sort(col('brq_m'), ascending = True).cache()
m_dist.show(20,0)

#m_dist.coalesce(1).write.csv('s3a://ada-dev/azeef/projects/'+COUNTRY+'/202110/smart_dualsim/'+MONTH+'/rfm/dist_test/m_dist', header = True, mode = 'overwrite')


##########################
#   Percentile ranking   #
##########################
r_window = Window.orderBy(F.col('last_seen').desc())
f_window = Window.orderBy(F.col('usage_frequency'))
m_window = Window.orderBy(F.col('brq_m'))
rfm_score = rfm.withColumn('r_percentile',F.percent_rank().over(r_window)).withColumn('f_percentile',F.percent_rank().over(f_window)).withColumn('m_percentile',F.percent_rank().over(m_window))

## R,F,M scores
rfm_score = rfm_score.withColumn('r',F.col('r_percentile')*5.0)\
         .withColumn('f',F.col('f_percentile')*5.0)\
         .withColumn('m',F.col('m_percentile')*5.0).cache()

rfm_score.show(5,0)


# Sanity check
row1 = rfm_score.agg({"r": "max"}).collect()[0]
print(row1["max(r)"]) # 4.299126312069529
row1 = rfm_score.agg({"f": "max"}).collect()[0]
print(row1["max(f)"]) # 4.993380362297462
row1 = rfm_score.agg({"m": "max"}).collect()[0]
print(row1["max(m)"]) # 5.0

# Sanity count
rfm_score.select('ifa','carrier').count() # 959268
rfm_score.select('ifa','carrier').distinct().count() # 959268


########################
# Prep the ultimate DF #
########################
# Read rfm_output
df1 = rfm_score.select('ifa','carrier','r','f','m')
df1 = df1.withColumn('rfm', ( col('r') + col('f') + col('m') )).cache()
df1.show(5,0)

'''
+------------------------------------+--------+---+---+---+---+
|ifa                                 |carrier |r  |f  |m  |rfm|
+------------------------------------+--------+---+---+---+---+
|ccd3c83c-3ecb-4abb-85af-79dc80640e40|Metfone |0.0|0.0|0.0|0.0|
|3d94fb7e-8ec3-42b6-a84a-355534cbcc8b|Cellcard|0.0|0.0|0.0|0.0|
|e3114d77-37b2-460e-b064-545b736c8321|Smart   |0.0|0.0|0.0|0.0|
|8e404f0f-8a93-41d1-b7ea-dd50743825ea|Cellcard|0.0|0.0|0.0|0.0|
|f104437e-f4cc-476a-b43c-e95fa4f15b4a|Metfone |0.0|0.0|0.0|0.0|
+------------------------------------+--------+---+---+---+---+
'''

###### ifa | carrier (array) | carrier_number
df_carrier = df1.groupBy('ifa').agg(F.collect_set('carrier').alias('carrier'))
df_carrier = df_carrier.withColumn('carrier_number', size(col('carrier')))
#df_carrier.show(5,0)
#df_carrier.printSchema()
df_carrier.select('carrier_number').distinct().show(10,0)

# Filter only carrier_number > 1
df_carrier = df_carrier.filter(col('carrier_number') > 1)
df_carrier = df_carrier.cache()
df_carrier.select('ifa').distinct().count() # 361155
df_carrier.select('ifa').count() # 361155
df_carrier.show(5,0)
'''
+------------------------------------+-------------------+--------------+
|ifa                                 |carrier            |carrier_number|
+------------------------------------+-------------------+--------------+
|000035d7-2075-4d3e-bf05-255211e46d23|[Metfone, Cellcard]|2             |
|00005e0e-49af-4945-a629-443d45958a0b|[Metfone, Smart]   |2             |
|000089d5-ab91-4635-8dc0-2d80d8f00bf1|[Metfone, Smart]   |2             |
|0000b396-e13d-425f-b63a-c7486a4c01bc|[Metfone, Smart]   |2             |
|0000bad0-6b67-4c29-b26d-0fb8c1ade14b|[Metfone, Smart]   |2             |
+------------------------------------+-------------------+--------------+
'''

###### ifa | all_rfm
df_all = df1.groupBy('ifa').agg(F.sum('rfm').alias('all_rfm'))
df_all = df_all.cache()
df_all.show(5,0)
'''
+------------------------------------+------------------+
|ifa                                 |all_rfm           |
+------------------------------------+------------------+
|054afad2-8246-45c1-9ccc-b347ac619be2|0.9284563661103812|
|5f37c0fc-9358-4424-86a3-4cb9b41cf327|9.629715139414287 |
|62870918-821e-448a-a3ef-e8fce10cfbce|9.958541609191103 |
|82c95efd-6aba-4783-bfa6-af40c09b3907|7.446775679302718 |
|8c2b9fb0-3ff2-4d5a-a0f5-66c065b6cbc3|4.082538027391879 |
+------------------------------------+------------------+
'''

###### ifa | carrier_rfm x3
list = ['Smart','Cellcard','Metfone']
# initialise df
df_init = df1.drop('r','f','m')

for i in list:
    print(''+i+'')
    df_init = df_init.withColumn(''+i+'_rfm', F.lit(0))
    df_init = df_init.withColumn(''+i+'_rfm', F.when(col('carrier') == ''+i+'', coalesce(df_init['rfm'],df_init[''+i+'_rfm'])).otherwise(0))

df_init.show(5,0)
#df_init.filter(col('carrier') == 'Smart').show(10,0)
#df_init.filter(col('carrier') == 'Cellcard').show(10,0)
#df_init.filter(col('rfm') != 0).show(10,0)

# Groupby ifa and Aggregate all columns accordingly
aggregate = ['Smart_rfm','Cellcard_rfm','Metfone_rfm']
exprs = [sum(x) for x in aggregate]
df_each = df_init.groupBy(['ifa']).agg(*exprs).sort(col('ifa'), descending = False)
df_each = df_each.cache()
df_each.show(5,0)
'''
+------------------------------------+------------------+------------------+------------------+
|ifa                                 |sum(Smart_rfm)    |sum(Cellcard_rfm) |sum(Metfone_rfm)  |
+------------------------------------+------------------+------------------+------------------+
|000035d7-2075-4d3e-bf05-255211e46d23|0.0               |3.3181091279768444|7.5254139409611005|
|00005e0e-49af-4945-a629-443d45958a0b|7.770692372076929 |0.0               |7.919452770283224 |
|000089d5-ab91-4635-8dc0-2d80d8f00bf1|6.334179295704179 |0.0               |1.0502772855011948|
|0000b396-e13d-425f-b63a-c7486a4c01bc|7.38471960497337  |0.0               |12.904308446219238|
|0000bad0-6b67-4c29-b26d-0fb8c1ade14b|10.869924810272188|0.0               |11.960111568837412|
+------------------------------------+------------------+------------------+------------------+
'''

# Rename aggregated columns
for i,var in zip(df_each.columns[1:],aggregate):
        df_each = df_each.withColumnRenamed(''+i+'', ''+var+'')

# change all column names to lowercase
df_each = df_each.select([F.col(x).alias(x.lower()) for x in df_each.columns])
df_each = df_each.cache()
df_each.show(5,0)
'''
+------------------------------------+------------------+------------------+------------------+
|ifa                                 |smart_rfm         |cellcard_rfm      |metfone_rfm       |
+------------------------------------+------------------+------------------+------------------+
|000035d7-2075-4d3e-bf05-255211e46d23|0.0               |3.3181091279768444|7.5254139409611005|
|00005e0e-49af-4945-a629-443d45958a0b|7.770692372076929 |0.0               |7.919452770283224 |
|000089d5-ab91-4635-8dc0-2d80d8f00bf1|6.334179295704179 |0.0               |1.0502772855011948|
|0000b396-e13d-425f-b63a-c7486a4c01bc|7.38471960497337  |0.0               |12.904308446219238|
|0000bad0-6b67-4c29-b26d-0fb8c1ade14b|10.869924810272188|0.0               |11.960111568837412|
+------------------------------------+------------------+------------------+------------------+
'''

###### Join all 3 subtables to create the ultimate DF and write
main_df = df_carrier.join(df_all, on='ifa', how='inner').join(df_each, on='ifa', how='inner')
main_df = main_df.cache()
main_df.show(5,0)
'''
+------------------------------------+-------------------+--------------+------------------+------------------+------------------+------------------+
|ifa                                 |carrier            |carrier_number|all_rfm           |smart_rfm         |cellcard_rfm      |metfone_rfm       |
+------------------------------------+-------------------+--------------+------------------+------------------+------------------+------------------+
|000035d7-2075-4d3e-bf05-255211e46d23|[Metfone, Cellcard]|2             |10.843523068937944|0.0               |3.3181091279768444|7.5254139409611005|
|00005e0e-49af-4945-a629-443d45958a0b|[Metfone, Smart]   |2             |15.690145142360151|7.770692372076929 |0.0               |7.919452770283224 |
|000089d5-ab91-4635-8dc0-2d80d8f00bf1|[Metfone, Smart]   |2             |7.384456581205374 |6.334179295704179 |0.0               |1.0502772855011948|
|0000b396-e13d-425f-b63a-c7486a4c01bc|[Metfone, Smart]   |2             |20.28902805119261 |7.38471960497337  |0.0               |12.904308446219238|
|0000bad0-6b67-4c29-b26d-0fb8c1ade14b|[Metfone, Smart]   |2             |22.8300363791096  |10.869924810272188|0.0               |11.960111568837412|
+------------------------------------+-------------------+--------------+------------------+------------------+------------------+------------------+
'''

##############
# Get Delta #
##############
# Add primary/secondary column
df1 = main_df.withColumn('primary_sim', F.when( (col('smart_rfm') > col('metfone_rfm')) & (col('smart_rfm') > col('cellcard_rfm')), F.lit('Smart') ).when( (col('metfone_rfm') > col('smart_rfm')) & (col('metfone_rfm') > col('cellcard_rfm')), F.lit('Metfone') ).when( (col('cellcard_rfm') > col('smart_rfm')) & (col('cellcard_rfm') > col('metfone_rfm')), F.lit('Cellcard') ).otherwise(0)).cache()

df1.show(5,0)

df2 = df1.withColumn('third_sim', F.when( (col('smart_rfm') < col('metfone_rfm')) & (col('smart_rfm') < col('cellcard_rfm')), F.lit('Smart') ).when( (col('metfone_rfm') < col('smart_rfm')) & (col('metfone_rfm') < col('cellcard_rfm')), F.lit('Metfone') ).when( (col('cellcard_rfm') < col('smart_rfm')) & (col('cellcard_rfm') < col('metfone_rfm')), F.lit('Cellcard') ).otherwise(0)).cache()

df2.show(5,0)

non_metfone = ['Smart','Cellcard']
non_cellcard = ['Smart','Metfone']
non_smart = ['Cellcard','Metfone']

df3 = df2.withColumn('secondary_sim', F.when( (col('primary_sim').isin(non_metfone)) & (col('third_sim').isin(non_metfone)), F.lit('Metfone') ).when(  (col('primary_sim').isin(non_cellcard)) & (col('third_sim').isin(non_cellcard)), F.lit('Cellcard') ).when( (col('primary_sim').isin(non_smart)) & (col('third_sim').isin(non_smart)), F.lit('Smart') ).otherwise(0))

df3 = df3.withColumn('secondary_sim', F.when( (col('primary_sim').isNotNull()) & (col('secondary_sim') == 0), F.lit('others') ).otherwise(col('secondary_sim')) )

df3 = df3.withColumn('third_sim', F.when(col('carrier_number') == 2, 0).otherwise(col('third_sim'))).cache()
df3.show(5,0)

# Delta rfm
#df1 = df.withColumn('delta_rfm', (col('smart_rfm') - col('cellcard_rfm') - col('metfone_rfm')))
#df1 = df.withColumn('delta_rfm', F.when(col('carrier_number') == 3 ,(col('smart_rfm') - col('metfone_rfm')) )\
#        .otherwise((col('smart_rfm') - col('cellcard_rfm') - col('metfone_rfm'))))
#df1 = df1.cache()
#df1.show(5,0)

df4 = df3.withColumn('delta_rfm', F.when( (col('secondary_sim')  == 'Metfone') | (col('primary_sim') == 'Metfone'), (col('smart_rfm') - col('metfone_rfm'))).when( (col('secondary_sim')  == 'Cellcard') | (col('primary_sim') == 'Cellcard'), (col('smart_rfm') - col('cellcard_rfm')) ).otherwise((col('smart_rfm') - col('metfone_rfm') - col('cellcard_rfm'))))
df4 = df4.cache()
df4.show(5,0)


ms_df = df4.select('ifa', 'primary_sim', 'secondary_sim', 'third_sim', 'all_rfm', 'smart_rfm', 'metfone_rfm', 'cellcard_rfm', 'delta_rfm')
save_path = 's3a://smart-bucket/rfm/202111/'
ms_df.coalesce(1).write.csv(save_path, header=True, mode='overwrite')



############
# Analysis #
############
ms_df.select('ifa').distinct().count() # 361155
ms_df.select('ifa').count() # 361155

#### Count delta >0, <0, =0
over_0 = ms_df.filter(col('delta_rfm') > 0)
over_0.select('ifa').distinct().count() # 136325 #'Primary smart simmers'
print(136325/361155*100) # 40.73929476263654% 37.746950755215906
print(175673/422967*100) # August multisim (41.53)
print(92851/222097*100) # August biweekly-1 multisim (41.81)
print(95661/232639*100) # August biweekly-2 multisim (41.12)

under_0 = ms_df.filter(col('delta_rfm') < 0)
under_0.select('ifa').distinct().count() # 194388 #'Secondary smart simmers'
print(194388/361155*100) # 53.47482382910385% 53.82398139303069
print(245355/422967*100) # August multisim (58.01)
print(127421/222097*100) # August biweekly-1 multisim (57.37)
print(135129/232639*100) # August biweekly-2 multisim (58.09)

at_0 = ms_df.filter(col('delta_rfm') == 0)
at_0.select('ifa').distinct().count() # 30442 #'Primary/Secondary smart simmers'
print(30442/361155*100) # 5.785881408259611% 8.429067851753402
print(1939/422967*100) # August multisim (0.46)
print(1825/222097*100) # August biweekly-1 multisim (0.82)
print(1849/232639*100) # August biweekly-2 multisim (0.79)


############################################
# Primary/Secondary sim carrier breakdown  #
############################################

data_path = "s3a://ada-dev/fadzilah/smart/rfm_multisim/monthly/202108"
#data_path = "s3a://ada-dev/fadzilah/smart/rfm_multisim/biweekly/202108_1"
ms_df = spark.read.csv(data_path,header=True)

primary_sims = ms_df.groupBy('primary_sim').agg(countDistinct('ifa').alias('count')).sort(col('count'), ascending=False)
primary_sims.show(10,0)
secondary_sims = ms_df.groupBy('secondary_sim').agg(countDistinct('ifa').alias('count')).sort(col('count'), ascending=False)
secondary_sims.show(10,0)

+-----------+------+
|primary_sim|count |
+-----------+------+
|Smart      |175685|
|Metfone    |148930|
|Cellcard   |95730 |
|0          |2622  |
+-----------+------+
+-----------+-----+
|primary_sim|count|
+-----------+-----+
|Smart      |92863|
|Metfone    |75802|
|Cellcard   |50964|
|0          |2468 |
+-----------+-----+
+-----------+-----+
|primary_sim|count|
+-----------+-----+
|Smart      |95675|
|Metfone    |80460|
|Cellcard   |53994|
|0          |2510 |
+-----------+-----+


+-------------+------+
|secondary_sim|count |
+-------------+------+
|Metfone      |170324|
|Smart        |149844|
|Cellcard     |97523 |
|0            |5276  |
+-------------+------+
+-------------+-----+
|secondary_sim|count|
+-------------+-----+
|Metfone      |89176|
|Smart        |76900|
|Cellcard     |50681|
|0            |5340 |
+-------------+-----+
+-------------+-----+
|secondary_sim|count|
+-------------+-----+
|Metfone      |92500|
|Smart        |81754|
|Cellcard     |53162|
|0            |5223 |
+-------------+-----+



############################################
# Monthly/Biweekly multisim matched IFA  #
############################################
data_path = "s3a://ada-dev/fadzilah/smart/rfm_multisim/monthly/202108"
monthly = spark.read.csv(data_path,header=True)
data_path = "s3a://ada-dev/fadzilah/smart/rfm_multisim/biweekly/202108_1"
biweekly1 = spark.read.csv(data_path,header=True)
data_path = "s3a://ada-dev/fadzilah/smart/rfm_multisim/biweekly/202108_2"
biweekly2 = spark.read.csv(data_path,header=True)

# Match IFA
mb1 = monthly.join(biweekly1,on=['ifa'],how='inner')
mb1.count()
mb1.select("ifa").distinct().count()
mb2 = monthly.join(biweekly2,on=['ifa'],how='inner')
mb2.count()
mb2.select("ifa").distinct().count()

# Match IFA,Primary/Secondary sims
mb1 = monthly.join(biweekly1,on=['ifa','primary_sim', 'secondary_sim'],how='inner')
mb1.count()
mb1.select("ifa").distinct().count()
mb2 = monthly.join(biweekly2,on=['ifa','primary_sim', 'secondary_sim'],how='inner')
mb2.count()
mb2.select("ifa").distinct().count()


mb1 = monthly.join(biweekly1,on=['ifa','primary_sim'],how='inner')
mb1.count()
mb1.select("ifa").distinct().count()
mb2 = monthly.join(biweekly2,on=['ifa','primary_sim'],how='inner')
mb2.count()
mb2.select("ifa").distinct().count()

mb1 = monthly.join(biweekly1,on=['ifa','secondary_sim'],how='inner')
mb1.count()
mb1.select("ifa").distinct().count()
mb2 = monthly.join(biweekly2,on=['ifa','secondary_sim'],how='inner')
mb2.count()
mb2.select("ifa").distinct().count()




###########
