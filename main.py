import joblib
import json
import numpy as np
import os
import pandas as pd
import yaml

from dotenv import load_dotenv
from model.build_model import UCTModel
from sqlalchemy import create_engine
from utils import load_dataset, stage2_correction_json, expected_feature_order, features_pass, set_start_end_dates, setup_logger

load_dotenv()
logger = setup_logger("logs/stage2.log")

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if config["stage2_specifications"]["use_cpu"]:
    logger.info(f"Using CPU for Stage2 correction ...")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# db connection
user, password, host, port = [os.getenv(var) for var in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]]
dbname = config["data"]["db"]["dbname"]
schema = config["data"]["db"]["schema"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")

# time range for data that should be corrected
if config["data"]["tables"][0]["start_date"] is None or config["data"]["tables"][0]["end_date"] is None:
    start, end = set_start_end_dates(engine, schema)
    logger.info(f"Start: Latest S2-corrected measurement: {start}.")
    logger.info(f"End: Latest DWD measurement: {end}.")
    config["data"]["tables"][0]["start_date"] = start
    config["data"]["tables"][0]["end_date"] = end

# load data for stage2 correction
df = load_dataset(config, engine, logger)

# if df empty, exit script
if df.empty:
    logger.info(f"No data available for Stage2 correction in the specified time window. Exiting script.")
    exit()

# load scaler and model
logger.info(f"Loading scaler and model ...")
scaler = joblib.load(config["stage2_specifications"]["file_paths"]["scaler"]) 
model = UCTModel(input_dim=len(expected_feature_order), hidden_units=config["model"]["hidden_units"])
model.build((None, len(expected_feature_order)))
model.load_weights(config["stage2_specifications"]["file_paths"]["model"])

# rescale features and prepare input for model
logger.info(f"Preparing input for model ...")
X = df.copy()
X.drop(columns=["id", "quell_id", "messzeit_utc", "stage0", "stage1", "stage2"], inplace=True)

scale_cols = [col for col in expected_feature_order if col not in features_pass]
X_scaled = pd.DataFrame(scaler.transform(X[scale_cols].values), columns=scale_cols, index=X.index)
X_pass = X[features_pass]
# reassemble in the exact column order the model was trained on
X_input = pd.concat([X_scaled, X_pass], axis=1)[expected_feature_order].values 

# make predictions
y_pred = model.predict(X_input, batch_size=config["stage2_specifications"]["batch_size"])
assert len(y_pred) == len(df), "Number of predictions does not match number of rows in the dataset for Stage2 correction"

# prepare DataFrame for DB upload
df_db_correction = pd.DataFrame({
    "id": df["id"],
    "messzeit_utc": df["messzeit_utc"],
    "stage0": df["temperature_value"],
    "stage1": df["stage1"],
    "stage2": y_pred.flatten()
})

# save last corrected timestamp
last_corrected_timestamp = df_db_correction["messzeit_utc"].max() 

# assertions to ensure data integrity before DB upload
assert df_db_correction["id"].isna().sum() == 0, "There are missing values in the 'id' column, which is required for DB upload"

# update corrected_values with stage2 correction
df_db_correction["corrected_values"] = df_db_correction.apply(stage2_correction_json, axis=1)

# convert to json for db upload
df_db_correction["corrected_values"] = df_db_correction["corrected_values"].apply(lambda x: json.dumps(x))
df_db_correction = df_db_correction[["id", "corrected_values"]]
df_db_correction['corrected_values'] = df_db_correction['corrected_values'].str.replace('NaN', 'null')

logger.info(f"Last corrected timestamp: {last_corrected_timestamp}")
logger.info(f"Uploading Stage2 corrections to database: {len(df_db_correction)} rows ...")

sql_update = f"""
UPDATE {schema}.messungen
SET corrected_values = %s
WHERE id = %s;
"""

# update db with corrected values
update_list = []
for index, row in df_db_correction.iterrows():
    update_list.append((row['corrected_values'], row['id']))

conn = engine.raw_connection()
cur = conn.cursor()

for update in update_list:
    cur.execute(sql_update, update)

# Update timestamp of last correction in db
sql_update_timestamp = f"""
UPDATE {schema}.control
SET timestamp_utc = '{last_corrected_timestamp}'
WHERE control_type = 'stage2_correction';
"""

cur.execute(sql_update_timestamp)

conn.commit()
cur.close()