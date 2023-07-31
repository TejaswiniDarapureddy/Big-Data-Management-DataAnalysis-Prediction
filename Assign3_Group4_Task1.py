#import all the libraries required for this task 
from pyspark.sql import SparkSession,DataFrame
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import*
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

#create a spark session if there is no existing one and give an application name for this task like below
spark = SparkSession.builder.appName("Assign3_Group4_Task1").getOrCreate()

#read the csv file using spark ans stored it in the DataFrame
data = spark.read.csv('hdfs://hadoop-nn001.cs.okstate.edu:9000/user/sdarapu/Group4_DataSet/IndianUniversityRankingFrom2017to2021.csv',header=True,inferSchema=True)

#check if there are any null values in dataframe
data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns]).show()

#function which takes input of mentioned columns fom csv and uses string indexer encodes a string column of labels to a column of label indices
def getIndex(ar_1, ar_2):
  return StringIndexer(inputCol=ar_1, outputCol=ar_2)

#converting string columns to indixces labels using string Indexers
InsID_indexer=getIndex("Institute ID", "INSTITUTE ID" )
data=InsID_indexer.fit(data).transform(data)

Name_indexer= getIndex("Name", "NAME" )
data=Name_indexer.fit(data).transform(data)

State_indexer= getIndex("State", "STATE" )
data=State_indexer.fit(data).transform(data)

City_indexer= getIndex("City", "CITY" )
data=City_indexer.fit(data).transform(data)

#convert spark dataframe to pandas dataframe
df=data.toPandas()

#replace Score column values with -1 which are out of specified range
df['Score'] = np.where((df['Score']>=65)|(df['Score']<=35),-1,df['Score'])

#replace NaN values in PR Rank column with -1
df['PR Rank'] = df['PR Rank'].replace(np.nan,-1)

#recheck if there are any null values and store it in another dataframe
df2=df[df.isna().any(axis=1)]

#extract the rows where the columns in a row contain -1 and store it in another dataframe
df_null = df[df.values == -1]

#drop the indexes from the original dataframe where they already existsed in df_null
df.drop(df_null.index, axis = 0, inplace = True)

#calculate cosine similarity between the null values dataframe and original dataframe
cosinesimilarity = cosine_similarity(df_null,df)
#printing cosine similarity length which is used below
#print(len(cosinesimilarity))

#library Return a new array of given shape and type, without initializing entries.
from numpy.ma.core import empty

#initialize a new list and take variable as 0
rows_list = []
c = 0
## method checks the similarity
def checkSim(c_s, index):
  ## returns the maximum similarity values by iterating through indexes
  return np.where(c_s[index]==np.max(c_s[index]))  

#iterating all the values in the cosine similarity values
while c < len(cosinesimilarity):  
  #pass the values to above function and store the value in variable
  max_val = checkSim(cosinesimilarity, c)
  #check the value is empty or not
  if max_val[0][0] is not empty :
    #append those values into one list
    rows_list.append(max_val[0][0])
  #increment the values to go to next value
  c = c+1
#print the values in list
#print(rows_list)

#place all the indexes in df_null to the variable
rplcr_list = df_null.index
#transform it to a list and assign to another list
list_replacer = rplcr_list.tolist()
#print those values in a variable
#print(list_replacer)

#intialize the values to 0
i = 0
j=0
indexx = 0
#loop which citerates the values in list_replacer
while indexx < len(list_replacer):
  #getting the indexes of rows containing null values along with columns
  col_tp = df_null[df_null.index == list_replacer[indexx]]
  #Identifying and storimg column names to be replaced
  colname = (col_tp.columns[(col_tp == -1).iloc[0]]).tolist()[0] 
  #fetching the index value from clean dataset and stores in f
  f=rows_list[i]
  #fetching the value at particular row and obtained column and store in h
  h=df.at[f,colname]
  #place the value in dataframe contain null and column
  df_null.at[list_replacer[indexx],colname]=h
  #fetch the inserted row from df_null and store it in another dataframe
  d2=df_null.iloc[j]
  #append all the newly inserted rows to original dataframe
  df=df.append(d2,ignore_index= True)
  #incrementing the values to get correct indexes
  indexx = indexx +1
  i=i+1
  j=j+1

#print the values upto 100
print(df.head(100))

#changing pandas dataframe to spark dataframe
sparkdf=spark.createDataFrame(df)

#last check if there are any null values in spark dataframe
sparkdf.select([count(when(col(c).isNull(), c)).alias(c) for c in sparkdf.columns]).show()

#storing spark dataframe as CSV into the specified path 
sparkdf.coalesce(1).write.mode("overwrite").option("header","true").csv("hdfs:////user/sdarapu/Assign3_Group4_Task1_Output")

#storing spark dataframe as CSV into the specified path and give this output file as input file for remaining Tasks 2-4
sparkdf.coalesce(1).write.option("header","true").csv("hdfs:////user/sdarapu/Assign3_Group4_Task1_Output_inpfor_Task2-4")









