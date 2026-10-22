import logging
import numpy as np
import pandas as pd
import sys

from logging.handlers import TimedRotatingFileHandler
from preprocessing.extract_JSON import extract_measurement_data_from_json, extract_additonal_parameters_from_json, extract_quell_properties, extract_raster_properties
from preprocessing.neighboring_senseBoxes import determine_neighboring_senseBoxes
from preprocessing.DWD_10min import attach_dwd_data_10min
from preprocessing.NaN_incorrect_values import nan_handling
from preprocessing.feature_engineering import messzeit_utc_handling, transform_categorical_features


# Orders of features must match the order used during model training!
expected_feature_order = [
    'temperature_value_neighbor_1', 
    'humidity_value_neighbor_1', 
    'pressure_value_neighbor_1', 
    'lux_value_neighbor_1', 
    'uv_value_neighbor_1', 
    'temperature_value_neighbor_2', 
    'humidity_value_neighbor_2', 
    'pressure_value_neighbor_2', 
    'lux_value_neighbor_2', 
    'uv_value_neighbor_2', 
    'temperature_value_DWD001',
    'temperature_value', 
    'humidity_value', 
    'pressure_value', 
    'lux_value', 
    'uv_value',
    'NDVI', 
    'GEB_D', 
    'Albedo',
    'BO_FEU', 
    'STRA_D', 
    'GEL_EXP', 
    'GEL_HOE', 
    'GEL_NEI', 
    'PARKP_D',
    'BESCHIRM', 
    'GEL_FORM', 
    'WASSER_D', 
    'GRU_VOL_D', 
    'VERS_2025', 
    'VEG_HOE',
    'LT_4', 
    'LT_14', 
    'hour_sin', 
    'hour_cos', 
    'doy_cos'
]

# These features should not be scaled
features_pass = [
    'GEL_FORM',
    'hour_sin', 
    'hour_cos', 
    'doy_cos'
]

def load_data(engine, dbconfig, table_list, logger):
    logger.info(f"Loading data from {table_list[0]['start_date']} to {table_list[0]['end_date']} ...")
    dbname = dbconfig["dbname"]
    schema = dbconfig["schema"]

    dfs = {}
    for table in table_list:
        name = table["name"]
        columns = table.get("columns", [])
        column_str = ", ".join(columns) if columns else "*"

        start = table.get("start_date")
        end = table.get("end_date")
        quell_typ = table.get("quell_typ")
        typ = table.get("typ")

        # WHERE-conditions
        conditions = []
        # time range condition for 'messzeit_utc' in table 'messungen'
        if name == "messungen" and (not columns or "messzeit_utc" in columns):
            if start and end:
                conditions.append(f"messzeit_utc BETWEEN '{start}' AND '{end}'")
            elif start:
                conditions.append(f"messzeit_utc >= '{start}'")
            elif end:
                conditions.append(f"messzeit_utc <= '{end}'")

        # Bedingung für 'quell_typ' in Tabelle 'messungen'
        if name == "messungen" and quell_typ:
            if isinstance(quell_typ, list) and len(quell_typ) > 0:
                joined = ", ".join([f"'{qt}'" for qt in quell_typ])
                conditions.append(f"quell_typ IN ({joined})")

        # exclude fport 10 in 'messungen' table
        if name == "messungen":
            conditions.append("(additional_parameters->'lora'->>'fport' IS NULL OR (additional_parameters->'lora'->>'fport')::integer <> 10)")

        # condition for 'typ' in table 'quelle'
        if name == "quelle" and typ is not None:
            if isinstance(typ, list) and len(typ) > 0:
                joined = ", ".join([f"'{t}'" for t in typ])
                conditions.append(f"typ IN ({joined})")

        # condition for 'id' in table 'location'
        if name == "location":
            conditions.append("id >= 100000 AND id < 200000")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        full_table_name = f"{schema}.{name}"

        query = f"SELECT {column_str} FROM {full_table_name} {where_clause};"

        try:
            dfs[name] = pd.read_sql_query(query, con=engine)
            logger.info(f"   Tabelle '{name}' geladen: {len(dfs[name])} Zeilen")
        except Exception as e:
            logger.error(f"   Fehler beim Laden von {name}: {e}")

    # merge 
    if "quelle" in dfs and "location" in dfs:
        try:
            dfs["quelle"] = pd.merge(dfs["quelle"], dfs["location"], left_on="raster_id", right_on="id", how="left")
            dfs["quelle"] = dfs["quelle"].drop(columns=["id_y"])
            dfs["quelle"] = dfs["quelle"].rename(columns={"id_x": "id", "properties": "raster_properties"})
            print(f"   DataFrames 'quelle' und 'location' erfolgreich zusammengeführt.")
        except Exception as e:
            print(f"   Fehler beim Zusammenführen der DataFrames 'quelle' und 'location': {e}")

    # merge messungen and quelle
    if "messungen" in dfs and "quelle" in dfs:
        try:
            if dbname == "UCT":
                dfs["quelle"] = dfs["quelle"][["id", "from_date", "until_date", "location_features", "quell_properties"]]
            elif dbname == "UrbanClimateTwin":
                dfs["quelle"] = dfs["quelle"][["id", "from_date", "until_date", "quell_properties", "location_id", "raster_properties"]]
            else:
                raise ValueError(f"Unknown dbname: {dbname}")
            
            # merge based on 'quell_id' and 'id'
            merged_df = pd.merge(dfs["messungen"], dfs["quelle"], left_on="quell_id", right_on="id", how="left")
            if "from_date" in merged_df.columns and "until_date" in merged_df.columns:
                
                # Keep only rows that meet the conditions
                merged_df = merged_df[
                    (merged_df["messzeit_utc"] >= merged_df["from_date"]) &
                    (merged_df["messzeit_utc"] <= merged_df["until_date"])
                ]

            if "location_id_x" in merged_df.columns and "location_id_y" in merged_df.columns:
                assert all(merged_df["location_id_x"] == merged_df["location_id_y"]), "Mismatch between location_id_x and location_id_y after merge"
                merged_df = merged_df.drop(columns=["location_id_y"])
                merged_df = merged_df.rename(columns={"location_id_x": "location_id"})

            if dbname == "UrbanClimateTwin" and "location_center" in merged_df.columns and "location" not in merged_df.columns:
                merged_df = merged_df.rename(columns={"location_center": "location"})
            
            logger.info(f"   DataFrames erfolgreich zusammengeführt.")
            merged_df = merged_df.drop(columns=["id_y"])
            merged_df = merged_df.rename(columns={"id_x": "id"})
        except Exception as e:
            logger.error(f"   Fehler beim Zusammenführen der DataFrames: {e}")
    
    logger.info(f"   Shape: {merged_df.shape}")	
    return merged_df if "messungen" in dfs and "quelle" in dfs else dfs.get("messungen", None)	    

def load_dataset(config, engine, logger):
    """
    Load datasets based on the provided configuration.
    Retrieves data from the database, applies preprocessing steps, and prepares the datasets for inference.
    """

    df = load_data(engine, config["data"]["db"], config["data"]["tables"], logger)
    df = df.sort_values(by=["messzeit_utc"])

    # Neighboring senseBoxes
    if config["preprocessing"].get("neighboring_senseBoxes", False):
        df = determine_neighboring_senseBoxes(df, 
                                              config["preprocessing"].get("number_of_neighbors", 0), 
                                              config["preprocessing"].get("measured_values_neighbors"), 
                                              config["preprocessing"].get("plausibility_checks"),
                                              logger,
                                              config["preprocessing"].get("neighbors_reduction_method"),
                                              config["preprocessing"].get("consistency_threshold_neighbors", None),
                                              config["preprocessing"].get("use_climatological_neighbors", False))

    # DWD data    
    if config["preprocessing"].get("DWD_10min", False):
        df =  attach_dwd_data_10min(df, config["preprocessing"]["DWD_stations_10min"], config["preprocessing"]["DWD_measured_values_10min"], logger)

    df = df[df["quell_typ"]== "senseBox"]

    # JSON data
    logger.info(f"Extracting JSON data...")
    df = extract_measurement_data_from_json(df, config["preprocessing"]["SB_measured_values"])
    df = extract_additonal_parameters_from_json(df, config["preprocessing"]["additional_parameters"])
    df = extract_quell_properties(df, config["preprocessing"]["quell_properties"])
    df = extract_raster_properties(df, config["preprocessing"]["raster_properties"])
    logger.info(f"   Shape: {df.shape}")

    drop_cols = config["preprocessing"].get("drop_columns", [])
    drop_cols = drop_cols + [col for col in df.columns if col.endswith("__OLD")] # drop old neighbor columns after plausibility check
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop inconsistent neighbors compared to DWD
    if config["preprocessing"]["consistency_threshold_neighbors"]:
        logger.info(f"Filtering inconsistent neighboring senseBoxes based on DWD measurements ...")
        df = filter_inconsistent_neighbors_DWD(df, config["preprocessing"]["consistency_threshold_neighbors"], logger)

    # NaN-handling
    df = nan_handling(df, config["preprocessing"]["NaN_handling"], logger)

    # Feature Engineering
    logger.info(f"Applying feature engineering...")
    df = transform_categorical_features(df)
    df = messzeit_utc_handling(df, config["preprocessing"]["feature_engineering"]["measurement_time_handling"])
    logger.info(f"   Shape: {df.shape}")

    # Extract stages from corrected_values JSON
    logger.info(f"Extracting previous stages from corrected_values ...")
    df[["stage0", "stage1", "stage2"]] = df.apply(extract_corrected_values, axis=1)
    df = df.drop(columns=["corrected_values"])
    logger.info(f"   Shape: {df.shape}")

    # Keep only data for which stage2 has not been applied yet
    logger.info(f"Discarding rows where stage2 is already applied ...")
    df = df[df["stage2"].isna()]
    logger.info(f"   Shape: {df.shape}")

    # convert to float64
    cols_to_exclude = ['id', 'quell_id', 'messzeit_utc']
    cols_to_convert = [col for col in df.columns if col not in cols_to_exclude]
    df[cols_to_convert] = df[cols_to_convert].astype(np.float64)
    df.reset_index(drop=True, inplace=True)

    return df

def extract_corrected_values(row):
    if isinstance(row["corrected_values"], dict) and "temperature" in row["corrected_values"]:
        temp_dict = row["corrected_values"]["temperature"]
        previous_stage_0 = temp_dict.get("previous", {}).get("stage_0", None)
        previous_stage_1 = temp_dict.get("previous", {}).get("stage_1", None)
        previous_stage_2 = temp_dict.get("previous", {}).get("stage_2", None)
        return pd.Series([previous_stage_0, previous_stage_1, previous_stage_2])
    else:
        return pd.Series([None, None, None])

def stage2_correction_json(row):
    json_dict = {
        "temperature": {
            "main": {
                "value": row["stage2"],
                "stage": 2
            },
            "previous": {
                "stage_0": row["stage0"],
                "stage_1": row["stage1"],
                "stage_2": row["stage2"]
            }
        }
    }
    return json_dict

def set_start_end_dates(engine, schema):
    # start: get latest timestamp from control table
    select_query = f"SELECT timestamp_utc FROM {schema}.control WHERE control_type = 'stage2_correction';"
    start_timestamp = pd.read_sql(select_query, engine).iloc[0, 0]

    # end: get latest timestamp for DWD
    select_query = f"SELECT MAX(messzeit_utc) FROM {schema}.messungen WHERE quell_id = 'DWD001';"
    end_timestamp = pd.read_sql(select_query, engine).iloc[0, 0]

    # return start and end dates
    start = start_timestamp.tz_localize('UTC').strftime("%Y-%m-%d %H:%M:%S")
    end = end_timestamp.tz_localize('UTC').strftime("%Y-%m-%d %H:%M:%S")
    return start, end

def filter_inconsistent_neighbors_DWD(df, threshold, logger):
    neighbor_cols = [col for col in df.columns if col.startswith("temperature_value_neighbor_")]
    if threshold is not None and "temperature_value_DWD001" in df.columns:
        for col in neighbor_cols:
            mask_consistency_dwd = abs(df["temperature_value_DWD001"] - df[col]) < threshold
            logger.info(f"   Removing rows where temperature difference between DWD and {col} exceeds consistency threshold of {threshold}°C: {(~mask_consistency_dwd).sum()} rows removed")
            df = df[mask_consistency_dwd] 
    
    logger.info(f"   Shape: {df.shape}")
    return df

def setup_logger(logfile : str, logname : str = None, level: int = 10, use_email: bool = False):
    """
    Logger initialisieren und konfigurieren.

    Args:
        logfile (str): Pfad zur Logdatei.
        logname (str): Name des Loggers.
        level (int): Logging-Level (z.B. logging.DEBUG (10), logging.INFO (20)). Standard ist 10.
    """
    logger = logging.getLogger(logname)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handlers = [
        TimedRotatingFileHandler(logfile, when="midnight", interval=1, backupCount=365, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.StreamHandler()
    ]
    
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    sys.excepthook = log_exception # log stderr 
    return logger

def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Let Ctrl+C interrupts pass through
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))