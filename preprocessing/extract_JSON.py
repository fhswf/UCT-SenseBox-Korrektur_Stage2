def extract_measurement_data_from_json(df, measured_values):
    for key, subkeys in measured_values.items():
        for subkey in subkeys:
            column_name = f"{key}_{subkey}"
            df[column_name] = df.apply(
                lambda row: row["measurement_data"].get(key, {}).get(subkey) if isinstance(row["measurement_data"], dict) else None,
                axis=1
            )
    return df.drop(columns=["measurement_data"])

def extract_additonal_parameters_from_json(df, additional_parameters):
    for key, subkeys in additional_parameters.items():
        for subkey in subkeys:
            column_name = f"{key}_{subkey}"
            df[column_name] = df.apply(
                lambda row: row["additional_parameters"].get(key, {}).get(subkey) if isinstance(row["additional_parameters"], dict) else None,
                axis=1
            )
    return df.drop(columns=["additional_parameters"])

def extract_quell_properties(df, quell_properties):
    for property_name in quell_properties:
        df[property_name] = df.apply(
            lambda row: row["quell_properties"].get(property_name) if isinstance(row["quell_properties"], dict) else None,
            axis=1
        )
    return df.drop(columns=["quell_properties"])

def extract_raster_properties(df, raster_properties):
    for property_name in raster_properties:
        df[property_name] = df.apply(
            lambda row: row["raster_properties"].get(property_name) if isinstance(row["raster_properties"], dict) else None,
            axis=1
        )
    return df.drop(columns=["raster_properties"])