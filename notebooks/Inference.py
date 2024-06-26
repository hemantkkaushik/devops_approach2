# Databricks notebook source
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Imports
from MLCORE_SDK import mlclient
import ast
from pyspark.sql import functions as F
from datetime import datetime
from delta.tables import *
import time
import pandas as pd
import mlflow
import pickle

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Model & Table parameters
# MAGIC

# COMMAND ----------

import yaml
try :
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = yaml.safe_load(solution_config)
except :
   pass
file_path = '/Workspace/Users/maheswar.chittib@tigeranalytics.com/.bundle/classification_dab/dev/files/data_config/SolutionConfig.yaml'

# Load the JSON file
with open(file_path, 'r') as file:
    solution_config = yaml.safe_load(file)

print(solution_config)

# COMMAND ----------

# GENERAL PARAMETERS
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config[f'sdk_session_id_{env}']
if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = sdk_session_id
use_latest = True

# JOB SPECIFIC PARAMETERS FOR INFERENCE
transformed_features_path = solution_config["inference"]["transformed_features_path"]+sdk_session_id
ground_truth_path = solution_config["inference"]["ground_truth_path"]+sdk_session_id
output_table_name = solution_config["inference"]["output_table_name"]+sdk_session_id
batch_size = int(solution_config["inference"].get("batch_size",500))
cron_job_schedule = solution_config["inference"].get("cron_job_schedule","0 */10 * ? * *")
model_name = solution_config["inference"]["model_name"]+sdk_session_id
model_version = solution_config["inference"]["model_version"]
primary_keys = solution_config["train"]["primary_keys"]
features = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]

# COMMAND ----------

if use_latest:
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    x = client.get_latest_versions(model_name)
    model_version = x[0].version

# COMMAND ----------

pickle_file_path = f"/mnt/FileStore/{db_name}"
dbutils.fs.mkdirs(pickle_file_path)
print(f"Created directory : {pickle_file_path}")
pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_name}.pickle"

# COMMAND ----------

try : 
  with open(pickle_file_path, "rb") as handle:
      obj_properties = pickle.load(handle)
      print(f"Instance loaded successfully")
except Exception as e:
  print(f"Exception while loading cache : {e}")
  obj_properties = {}
print(f"Existing Cache : {obj_properties}")

# COMMAND ----------

if not obj_properties :
  start_marker = 1
elif obj_properties and obj_properties.get("end_marker",0) == 0:
  start_marker = 1
else :
  start_marker = obj_properties["end_marker"] + 1
end_marker = start_marker + batch_size - 1

print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

features_df = spark.sql(f"SELECT * FROM {db_name}.{transformed_features_path}")
gt_data = spark.sql(f"SELECT * FROM {db_name}.{ground_truth_path}")

# COMMAND ----------

FT_DF = features_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))
GT_DF = gt_data.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

if not FT_DF.first():
  dbutils.notebook.exit("No data is available for inference, hence exiting the notebook")

# COMMAND ----------

mlclient.log(
        operation_type="job_run_add", 
        session_id = sdk_session_id, 
        dbutils = dbutils, 
        request_type = "inference" , 
        job_config = 
        {
            "table_name" : output_table_name,
            "table_type" : "Inference_Output",
            "batch_size" : batch_size,
            "output_table_name" : output_table_name,
            "model_name" : model_name,
            "model_version" : model_version,
            "feature_table_path" : f"{db_name}.{transformed_features_path}",
            "ground_truth_table_path" : f"{db_name}.{ground_truth_path}",
            "env" : env,
            "quartz_cron_expression" : cron_job_schedule
        },
        verbose = True,
        spark = spark
        )

# COMMAND ----------

ground_truth = gt_data.toPandas()[primary_keys + target_columns]
tranformed_features_df = FT_DF.toPandas()
tranformed_features_df.dropna(inplace=True)
tranformed_features_df.shape

# COMMAND ----------

FT_DF = spark.createDataFrame(tranformed_features_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Load Inference Features Data

# COMMAND ----------

inference_df = tranformed_features_df[features]
display(inference_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Model

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

predictions = loaded_model.predict(inference_df)
type(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Output Table

# COMMAND ----------

tranformed_features_df["prediction"] = predictions
tranformed_features_df = pd.merge(tranformed_features_df,ground_truth, on=primary_keys[0], how='left')
output_table = spark.createDataFrame(tranformed_features_df)

# COMMAND ----------

output_table = output_table.withColumnRenamed(target_columns[0],"ground_truth_value")
output_table = output_table.withColumn("acceptance_status",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_time",F.lit(None).cast("long"))
output_table = output_table.withColumn("accepted_by_id",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_by_name",F.lit(None).cast("string"))
output_table = output_table.withColumn("moderated_value",F.lit(None).cast("double"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Save Output Table

# COMMAND ----------

output_table.createOrReplaceTempView(output_table_name)

output_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == output_table_name.lower() and not table_data.isTemporary]

if not any(output_table_exist):
  print(f"CREATING TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{output_table_name} AS SELECT * FROM {output_table_name}")
else :
  print(F"UPDATING TABLE")
  spark.sql(f"INSERT INTO hive_metastore.{db_name}.{output_table_name} SELECT * FROM {output_table_name}");

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Output Table dbfs path

# COMMAND ----------

output_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{output_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(f"output Hive Path : {output_hive_table_path}")

# COMMAND ----------

feature_table_dbfs_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{transformed_features_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
gt_table_dbfs_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{ground_truth_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(feature_table_dbfs_path, gt_table_dbfs_path)

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Register Inference artifacts

# COMMAND ----------

mlclient.log(operation_type = "register_inference",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    output_table_name=output_table_name,
    output_table_path=output_hive_table_path,
    feature_table_path=feature_table_dbfs_path,
    ground_truth_table_path=gt_table_dbfs_path,
    model_name=model_name,
    model_version=model_version,
    num_rows=output_table.count(),
    cols=output_table.columns,
    table_type="internal",
    batch_size = batch_size,
    env = env,
    column_datatype = output_table.dtypes,
    table_schema = output_table.schema,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    # register_in_feature_store=True,
    verbose=True)

# COMMAND ----------

obj_properties['end_marker'] = end_marker
with open(pickle_file_path, "wb") as handle:
    pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Instance successfully saved successfully")
