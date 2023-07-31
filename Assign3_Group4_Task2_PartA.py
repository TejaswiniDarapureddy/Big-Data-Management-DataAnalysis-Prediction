#importing the required packages
## importing SParkSession SQLContext
from pyspark.sql import SparkSession,SQLContext
## SparkConf , SparkContext
from pyspark import SparkConf,SparkContext
## importing VectorAssembler
from pyspark.ml.feature import StringIndexer,VectorAssembler
## Importing Regreesion Evalutors
from pyspark.ml.evaluation import RegressionEvaluator
## importing minMaxScaler
from pyspark.ml.feature import MinMaxScaler 
## Importing udf
from pyspark.sql.functions import udf
## importing DoubleType
from pyspark.sql.types import DoubleType
## importing sql types
from pyspark.sql.types import *
## importing pipeline
from pyspark.ml import Pipeline
## vector to array
from pyspark.ml.functions import vector_to_array
## importing concat_ws and col
from pyspark.sql.functions import concat_ws,col
## importing LinearRegreesion library
from pyspark.ml.regression import LinearRegression
## Importing pd
import pandas as pd
#Creating the sparkconfiguration
## using sprkSession builder creating spark session
spark = SparkSession.builder.appName("Assignment3_Group4_Task2_PartA").getOrCreate()
#Reading the csv file along with the header data into a dataframe df
df = spark.read.csv("hdfs://hadoop-nn001.cs.okstate.edu:9000/user/sdarapu/Assign3_Group4_Task1_Output_inpfor_Task2-4/part-00000-571e77d2-85ae-4579-92f8-dd4dc788ab7f-c000.csv", header = True, inferSchema = True)
#printing the schema of the dataframe
df.printSchema()
#converting the String columns into double data type
#InsID_indexer=StringIndexer(inputCol="Institute ID", outputCol="INSTITUTE ID")
#df=InsID_indexer.fit(df).transform(df)
#Name_indexer=StringIndexer(inputCol="Name", outputCol="NAME")
#df=Name_indexer.fit(df).transform(df)
#State_indexer=StringIndexer(inputCol="State", outputCol="STATE")
#df=State_indexer.fit(df).transform(df)
#City_indexer=StringIndexer(inputCol="City", outputCol="CITY")
#df=City_indexer.fit(df).transform(df)
#priting the modified schema
#df.printSchema()

## using udf creating list unwrapper
UN_WRAPPER = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
#  colList
col_list = ["NAME","STATE","Score","PR Rank","INSTITUTE ID"]

indexx = 0
## iterating over a while loop

while indexx < len(col_list):
    ## creating vector Assembler .....
	assembler = VectorAssembler(inputCols=[col_list[indexx]],outputCol=col_list[indexx]+"_Vect")

	# MinMaxScaler Transformation

	scaler = MinMaxScaler(inputCol=col_list[indexx]+"_Vect", outputCol=col_list[indexx]+"_Scaled")

	# Pipeline of VectorAssembler and MinMaxScaler

	pipeline = Pipeline(stages=[assembler, scaler])

	# Fitting pipeline on dataframe

	df = pipeline.fit(df).transform(df).withColumn(col_list[indexx]+"_Scaled", UN_WRAPPER(col_list[indexx]+"_Scaled")).drop(col_list[indexx]+"_Vect")
    ## indexx incrementer
	indexx = indexx +1

#using vector assembler of coverting the columns into vectors that are used for prediction
vectorAssembler = VectorAssembler(inputCols = df.columns[9:], outputCol = 'features')
## setting params 
vectorAssembler.setParams(handleInvalid="skip")#transforming the data frame
transform_output=vectorAssembler.transform(df)
#selecting the features and Year column and storing into final_df dataframe
final_df=transform_output.select('features','Year')
#splitting the data into training and test data using randomSplit function
(trainingData, testData) = final_df.randomSplit([0.7, 0.3],80)

#printing the number of rows for training and testing data
print("Number of train records are",trainingData.count())
## print the count of the testdata
print("Number of test records are",testData.count())
## training Data
trainingData.describe().show()
## test Data display
testData.describe().show()
## linear regression model 
lr=LinearRegression(featuresCol = 'features', labelCol='Year')

## model fitting using the training Data
lr_model=lr.fit(trainingData)
## model co-eff
## co -effs from the model
c=round(lr_model.coefficients[0],2)
## getting values for s
s=round(lr_model.intercept,2)
## Printing the formula for Linear Regression
print(f"""the formula for linear regression is Year={c}*features+{s}""")
## Model for Linear regreesion predictions
lr_predictions = lr_model.transform(testData)
## putting the selctesd columns for the regression mdoel 
lr=lr_predictions.select("prediction","Year","features")
## displaying the first 100 results
lr.show(100)
### linear regression model with column -- features and vector array
lr = lr.withColumn('features', vector_to_array('features'))
#### Linear regression model column changing the vector to array and array to string using the concat
lr = lr.withColumn("features",concat_ws(",",col("features")))
### linear regression output being saved to the out_put directory
lr.coalesce(1).write.mode("overwrite").option("header","true").csv("hdfs:////user/sdarapu/Assign3_Group4_Task2_PartA_Output")




