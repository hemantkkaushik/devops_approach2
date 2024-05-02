# Databricks notebook source
# %pip install databricks-feature-store
%pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md <b> User Inputs

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
try :
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = yaml.safe_load(solution_config)
except :
   pass




#file_path = '../data_config/SolutionConfig.yaml'
file_path = '/Workspace/Users/maheswar.chittib@tigeranalytics.com/.bundle/classification_dab/dev/files/data_config/SolutionConfig.yaml'

# Load the JSON file
with open(file_path, 'r') as file:
    solution_config = yaml.safe_load(file)

print(solution_config)

# COMMAND ----------

from MLCORE_SDK import mlclient
from pyspark.sql import functions as F

# GENERAL PARAMETERS
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config[f'sdk_session_id_{env}']

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config[f'sdk_session_id_{env}']

env = solution_config['ds_environment']
db_name = sdk_session_id

# DE SPECIFIC PARAMETERS
primary_keys = solution_config["data_engineering"]["data_engineering_ft"]["primary_keys"]
features_table_name = solution_config["data_engineering"]["data_engineering_ft"]["features_table_name"] + sdk_session_id
features_dbfs_path = solution_config["data_engineering"]["data_engineering_ft"]["features_dbfs_path"]
batch_size = int(solution_config["data_engineering"]["data_engineering_ft"].get("batch_size",500))

# COMMAND ----------

from MLCORE_SDK import mlclient
mlclient.log(
    operation_type="job_run_add",
    session_id = sdk_session_id,
    dbutils = dbutils,
    request_type = "de",
    job_config =
    {
        "table_name" : features_table_name,
        "table_type" : "Source",
        "batch_size" : batch_size
    },
    verbose = True,
    spark = spark
    )

# COMMAND ----------

features_df = spark.read.load(features_dbfs_path)

# COMMAND ----------

features_df = features_df.drop('date','id','timestamp')

# COMMAND ----------

features_df.display()

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd",
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
features_df = features_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
features_df = features_df.withColumn("date", F.lit(date))
features_df = features_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in features_df.columns :
  window = Window.orderBy(F.monotonically_increasing_id())
  features_df = features_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

# COMMAND ----------

# DBTITLE 1,ADD A MONOTONICALLY INREASING COLUMN - "id"
features_df.createOrReplaceTempView(features_table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == features_table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{features_table_name} AS SELECT * FROM {features_table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO hive_metastore.{db_name}.{features_table_name} SELECT * FROM {features_table_name}");

# COMMAND ----------

from pyspark.sql import functions as F
features_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{features_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(f"Features Hive Path : {features_hive_table_path}")

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()


# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md <b> Use MLCore SDK to register Features and Ground Truth Tables

# COMMAND ----------

table_description = "This is the source table for a marketing campaign use case designed for a retail store, containing raw features such as purchases and customer details. It will be used for feature engineering to build a classification model in the marketing and customer segmentation domain tailored specifically for a retail environment. The table comprises 2240 records and 20 columns."

# COMMAND ----------

mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = features_table_name,
    num_rows = features_df.count(),
    cols = features_df.columns,
    column_datatype = features_df.dtypes,
    table_schema = features_df.schema,
    primary_keys = primary_keys,
    table_path = features_hive_table_path,
    table_type="internal",
    table_sub_type="Source",
    env = "dev",
    compute_usage_metrics = compute_metrics,
    table_description = table_description,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    # register_in_feature_store=True,
    verbose=True,)
