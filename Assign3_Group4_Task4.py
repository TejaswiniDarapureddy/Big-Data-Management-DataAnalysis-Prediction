### Importing all Necessary libraries
## Importing RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
## Importing sql types
from pyspark.sql.types import*
from pyspark.sql.functions import count,when,col
## Importing MinMaxScalar library
from pyspark.ml.feature import MinMaxScaler
## Importing Normalizer
from pyspark.ml.feature import Normalizer
## Importing udf
from pyspark.sql.functions import udf
## Importing SparkSession, DataFrame
from pyspark.sql import SparkSession,DataFrame
## Importing  StringIndexer,VectorAssembler
from pyspark.ml.feature import StringIndexer, VectorAssembler
## Importing Pipeline
from pyspark.ml import Pipeline
## Importing cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
## Importing pandas
import pandas as pd
## Importing numpy
import numpy as np
## Importing Double Type
from pyspark.sql.types import DoubleType
## Importing LinearRegression
from pyspark.ml.regression import LinearRegression

####################################################################################
##############  Pipeline for TASK -1 STARTS ####################################################
###################################################################################


## creating spark session
spark = SparkSession.builder.appName("Assignment3_Group4_Task4").getOrCreate()
## creating data frame using read csv
data = spark.read.csv('hdfs://hadoop-nn001.cs.okstate.edu:9000/user/sdarapu/Group4_DataSet/IndianUniversityRankingFrom2017to2021.csv',header=True,inferSchema=True)

data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns]).show()

## below function returns String indexer
def getIndex(ar_1, ar_2):
  return StringIndexer(inputCol=ar_1, outputCol=ar_2)

## Institute ID indexer
InsID_indexer=getIndex("Institute ID", "INSTITUTE ID"  )
#data=InsID_indexer.fit(data).transform(data)

## Name indexer
Name_indexer= getIndex("Name", "NAME" )
#data=Name_indexer.fit(data).transform(data)
## State Idexers
State_indexer= getIndex("State", "STATE" )
#data=State_indexer.fit(data).transform(data)
## City Indexer
City_indexer= getIndex("City", "CITY" )
#data=City_indexer.fit(data).transform(data)

## Pipeline1 using pipeline
pipeline1=Pipeline(stages=[InsID_indexer,Name_indexer,State_indexer,City_indexer])
## Pipeline fit
data=pipeline1.fit(data).transform(data)

## pandas Df conversion
df=data.toPandas()

## out for range checkin validation
df['Score'] = np.where((df['Score']>=65)|(df['Score']<=35),-1,df['Score'])
## replaceing Nan with -1
df['PR Rank'] = df['PR Rank'].replace(np.nan,-1)
## checking isNa
df2=df[df.isna().any(axis=1)]
## assing null values to -1
df_null = df[df.values == -1]
df.drop(df_null.index, axis = 0, inplace = True)

## Cosine similarity
cosinesimilarity = cosine_similarity(df_null,df)
## printing the length of cosine similarity
#print(len(cosinesimilarity))

## Import on empty
from numpy.ma.core import empty
## Empty row list
rows_list = []
c = 0
## method checks the similarity
def checkSim(c_s, index):
  ## returns the maximum sim
  return np.where(c_s[index]==np.max(c_s[index]))  
##iterating over a len of cosinesimilarity
while c < len(cosinesimilarity):  
  ## calling checkSim function
  max_val = checkSim(cosinesimilarity, c)
  ## chekcing whether its empty or not
  if max_val[0][0] is not empty :
    ## appending to rowsList
    rows_list.append(max_val[0][0])
  c = c+1

## Printing rows list
#print(rows_list)

## replacer using null df index
rplcr_list = df_null.index
list_replacer = rplcr_list.tolist()
## Printitn the list replacer
#print(list_replacer)

i=0
j=0
indexx = 0
## iterating over len of list iterator
while indexx < len(list_replacer):
  ## list replacer df null 
  col_tp = df_null[df_null.index == list_replacer[indexx]]
  #getting column name to be replaced
  colname = (col_tp.columns[(col_tp == -1).iloc[0]]).tolist()[0] 
  ## adding to row list
  f=rows_list[i]
  ## replacing at f
  h=df.at[f,colname]
  ## replacing at list_replacer inderxx
  df_null.at[list_replacer[indexx],colname]=h
  ## getting d2 from df_null
  d2=df_null.iloc[j]
  ## creating a df with appening all d2 elements from the loop
  df=df.append(d2,ignore_index= True)
  ## indexx incrementer -- 
  indexx = indexx +1
  i=i+1
  j=j+1
## c=converting to spark data frame
sparkdf=spark.createDataFrame(df)
#last check if there are any null values in spark dataframe
sparkdf.select([count(when(col(c).isNull(), c)).alias(c) for c in sparkdf.columns]).show()


#pipeline for Task2 and Task3 for Part A(Linear Regression)


################################################################################
##########  Pipeline for TASK -2 & TASK-3 for Linear Regression Starts #####################################################
################################################################################



#Reading the csv file along with the header data into a dataframe df
df1 = spark.read.csv("hdfs://hadoop-nn001.cs.okstate.edu:9000/user/sdarapu/Assign3_Group4_Task1_Output_inpfor_Task2-4/part-00000-571e77d2-85ae-4579-92f8-dd4dc788ab7f-c000.csv", header = True, inferSchema = True)
#printing the schema of the dataframe
df1.printSchema()
#Type_ChangeLst = udf(lambda x: list(x)[0])
Type_ChangeLst = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
## Printiing Type_ChangeLst
print(Type_ChangeLst)
## col_list defining 
col_list = ["NAME","STATE","Score","PR Rank","INSTITUTE ID"]
## Printing col_list
print(col_list)

indexx = 0

## Iterating over a while loop
while indexx < len(col_list):
  ## creating assembler using vector assembler
	assembler = VectorAssembler(inputCols=[col_list[indexx]],outputCol=col_list[indexx]+"_Vect")

	# MinMaxScaler Transformation

	scaler = MinMaxScaler(inputCol=col_list[indexx]+"_Vect", outputCol=col_list[indexx]+"_Scaled")

	# Pipeline of VectorAssembler and MinMaxScaler

	pipeline = Pipeline(stages=[assembler, scaler])

	# Fitting pipeline on dataframe
  ## creating data frame
	df1 = pipeline.fit(df1).transform(df1).withColumn(col_list[indexx]+"_Scaled", Type_ChangeLst(col_list[indexx]+"_Scaled")).drop(col_list[indexx]+"_Vect")
  ## Indexx incrementer 
	indexx = indexx +1

## printing schema 
df1.printSchema()
## vectorAssembler creation
vectorAssembler = VectorAssembler(inputCols = df1.columns[9:], outputCol = 'features')
## setting params to assembler
vectorAssembler.setParams(handleInvalid="skip")

## Normalizer using normalizer
normalizer=Normalizer(inputCol="features", outputCol="features_up", p=2)
## random Split
(trainingData, testData) = df1.randomSplit([0.7, 0.3],80)

## traning Data  cout
print("Number of train records are",trainingData.count())
## testDataCount
print("Number of test records are",testData.count())

## Linear Regreession model
lr = LinearRegression(featuresCol="features",labelCol='Year',)
## Pipline stages
pipeline = Pipeline(stages=[vectorAssembler,normalizer, lr])
## calling pipeline .fit
lr_model = pipeline.fit(trainingData)
## model trand=sformation
lr_predictions = lr_model.transform(testData)



################################################################################
########################## RMSE ################################################
################################################################################

## calling regression evalutior
evaluator = RegressionEvaluator(labelCol="Year", predictionCol="prediction", metricName="rmse")
## Printing RMSE 
print("------------------ RMSE for Linear Regression --------------------------------------------",evaluator.evaluate(lr_predictions))

## calling regression Evaluator
evaluator = RegressionEvaluator(labelCol="Year", predictionCol="prediction", metricName="r2")
## Printing R2
print("------------------ R2 for Linear Regression --------------------------------------------",evaluator.evaluate(lr_predictions))



################################################################################
##########  Pipeline for TASK -2 & TASK-3 for Random Forest Starts #####################################################
#############################################################################

#pipeline for Task2 and Task3 for Part B (Random Forest Regression)
## Importing library
from pyspark.ml.regression import RandomForestRegressor

## Calling vector assembler
vectorAssembler1 = VectorAssembler(inputCols = df1.columns[9:], outputCol = 'features')
## setting params for vector Assembler
vectorAssembler1.setParams(handleInvalid="skip")

## using normalizer
normalizer1=Normalizer(inputCol="features", outputCol="features_up", p=2)
## creating trainign Data and test Dtaa
(trainingData1, testData1) = df1.randomSplit([0.7, 0.3],80)
## coun tof data
print("Number of train records are",trainingData1.count())
## printing count of test Data
print("Number of test records are",testData1.count())

## calling Random Regreessior
rf = RandomForestRegressor(featuresCol="features",labelCol='Year',)
## calling pipeline 
pipeline = Pipeline(stages=[vectorAssembler1,normalizer1, rf])
## calling Pipeline fit
rf_model = pipeline.fit(trainingData1)
## rf_model tranform
rf_predictions = rf_model.transform(testData1)


## Calling Regression Evalutor
evaluator = RegressionEvaluator(labelCol="Year", predictionCol="prediction", metricName="rmse")
## Pritning RMSE 
print("------------------RMSE for Random Forest Regression --------------------------------------------",evaluator.evaluate(rf_predictions))

####  Caling Regression Evalutor
evaluator = RegressionEvaluator(labelCol="Year", predictionCol="prediction", metricName="r2")
## Printing R2
print("------------------R2 for Random Forest Regression --------------------------------------------",evaluator.evaluate(rf_predictions))


