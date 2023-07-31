## Importing the all libraries
## Importing Spark Session
from pyspark.sql import SparkSession
## Importing PCA  and Vector Assembler
from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler
## Importing SPark Conf and Spark COntext
from pyspark import SparkConf,SparkContext
## Importing SPark Session and SQL COntext and Data Frame
from pyspark.sql import SparkSession,SQLContext,DataFrame
## Spark StringIndexer, Vector Assembler
from pyspark.ml.feature import StringIndexer,VectorAssembler
## Importing MinMaxScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
## Installing Pipeline
from pyspark.ml import Pipeline
## importing udf
from pyspark.sql.functions import udf
## Importing DoubleType
from pyspark.sql.types import DoubleType

## sparkSession builder .. using sparkSession
spark = (SparkSession.builder.appName("Assign3_Group4_Task5").enableHiveSupport().getOrCreate())

## Below function returns the data Frame from the input path
## the arg -1 : spark session instance
## the arg_2 : path_to_input
def getDataFram(spark, path_to_input):
  ## returning the data_frame using spark.read.csv from the given input path
  ## ----return
  return spark.read.csv(path_to_input, header=True, inferSchema=True, mode="DROPMALFORMED", encoding='UTF-8')

## input path
path_to_input = 'hdfs://hadoop-nn001.cs.okstate.edu:9000/user/sdarapu/Assign3_Group4_Task1_Output_inpfor_Task2-4/part-00000-571e77d2-85ae-4579-92f8-dd4dc788ab7f-c000.csv'
## calling getDataFram function defined on above
## And Assigning to the data Frame
df = getDataFram(spark, path_to_input)

print("############### Original input _Data ####################")
## show the df data
df.show(truncate=False)

#converting Institute Id into label indices using string indexer and fit,transform
#InsID_indexer=StringIndexer(inputCol="Institute ID", outputCol="INSTITUTE ID")

#df=InsID_indexer.fit(df).transform(df)

#converting Name into label indices using string indexer and fit,transform

#Name_indexer=StringIndexer(inputCol="Name", outputCol="NAME")

#df=Name_indexer.fit(df).transform(df)

#converting State into label indices using string indexer and fit,transform

#State_indexer=StringIndexer(inputCol="State", outputCol="STATE")

#df=State_indexer.fit(df).transform(df)

#converting City into label indices using string indexer and fit,transform

#City_indexer=StringIndexer(inputCol="City", outputCol="CITY")

#df=City_indexer.fit(df).transform(df)

## Using udf rounding
un_wraper_li = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
## creating a col list
col_list = ["PR Score","PR Rank","Score"]
indexx = 0
## loop over a while loop
while indexx < len(col_list):
  ## getting assembler from vector assembler
  assembler = VectorAssembler(inputCols=[col_list[indexx]],outputCol=col_list[indexx]+"_Vect")
  # MinMaxScaler Transformation
  scaler = MinMaxScaler(inputCol=col_list[indexx]+"_Vect", outputCol=col_list[indexx]+"_Scaled")
  # Pipeline of VectorAssembler and MinMaxScaler
  pipeline = Pipeline(stages=[assembler, scaler])
  # Fitting pipeline on dataframe
  df = pipeline.fit(df).transform(df).withColumn(col_list[indexx]+"_Scaled", un_wraper_li(col_list[indexx]+"_Scaled")).drop(col_list[indexx]+"_Vect")  
  ## increamenting the loop indexx
  indexx = indexx +1

## df.show to check and display
df.show()

## below function return the assembler with transformation
def get_featureVec(df, assembler):

  ## returns the transformed assembler
  return assembler.transform(df)

## assigning the  assembler
assembler = VectorAssembler(inputCols=df.columns[9:], outputCol="variable")

## assigning to the featured vectors
vec_fetrd = get_featureVec(df, assembler)

## showing the featured vectors
vec_fetrd.show()

## function to return the modelFit
def getModelFit(scaler, vec_fetrd):
  ## returning the model fitted
  return scaler.fit(vec_fetrd)

## defining the scaler variable
scaler = StandardScaler(inputCol="variable", outputCol="Standardized variate", withStd=True, withMean=True)

## calling getModelFit function
## Assigning the result to scalModel
s_mdl = getModelFit(scaler, vec_fetrd)

#scaler.fit(vec_fetrd)
def getTrans(s_mdl, vec_fetrd):
  ## return the transformed model
  return s_mdl.transform(vec_fetrd)

## defining the   
VECTOR_STD = getTrans(s_mdl, vec_fetrd)

##s_mdl.transform(vec_fetrd)

print(" #################### Standardized data ####################")
## displaying the 
VECTOR_STD.select("Standardized variate").show(truncate=False)

### function to return the PCA
def getPCA(i_k, i_col, o_col ):
  ## returning the result
  return PCA(k=i_k, inputCol=i_col, outputCol=o_col)
## definign the variables
i_k=3
i_col="Standardized variate"
o_col="Main component score"

pca = getPCA(i_k, i_col, o_col)
##pca = PCA(k=3, inputCol="Standardized variate", outputCol="Main component score")

### function to getModelFit 
def getModelFit_2(pca, VECTOR_STD ):

  ## return fitted model
  return pca.fit(VECTOR_STD)

## assigning the resultant to a MDL_PCA
MDL_PCA = getModelFit_2(pca, VECTOR_STD )

print(" ########### Eigenvector ####")
### printting the MDL_PCA
print(MDL_PCA.pc)

## printing the contribution rate
print("################ Contribution rate #################")
####### printing the explaninedVariance
print(MDL_PCA.explainedVariance)

FINAL_PCA_SCORE = MDL_PCA.transform(VECTOR_STD).select("Main component score")
#########################################
## printing the final score
print("################ FINAL PCA SCOREs ################")
####### final_pca_score
FINAL_PCA_SCORE.show(truncate=False)

