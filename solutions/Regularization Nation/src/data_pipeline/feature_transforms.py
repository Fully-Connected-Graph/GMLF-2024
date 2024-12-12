import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def fill_nan(df):
    # Parse 'measurement_time' as datetime and set it as the index
    df["measurement_time"] = pd.to_datetime(df["measurement_time"])
    df.set_index("measurement_time", inplace=True)

    # Fill NaN values with values from exactly one week ago
    df = df.fillna(df.shift(168))  # 168 hours in a week

    # Reset the index back if needed
    df.reset_index(inplace=True)

    return df


def scale_dataframe(df, scaler_filename, target_scaler_filename):
    """
    Scale each column in the DataFrame using StandardScaler
    and save the scaler parameters to a file. The 'target' column
    is scaled separately and saved in a separate scaler.

    Args:
        df: pandas DataFrame to scale
        scaler_filename: where to save the scaler parameters for features
        target_scaler_filename: where to save the scaler parameters for the target column

    Returns:
        Scaled DataFrame
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Separate datetime, ID, and target columns if they exist
    if "measurement_time" in df_copy.columns:
        datetime_col = df_copy["measurement_time"]
        df_copy = df_copy.drop("measurement_time", axis=1)
    if "ID" in df_copy.columns:
        id_col = df_copy["ID"]
        df_copy = df_copy.drop("ID", axis=1)
    if "target" in df_copy.columns:
        target_col = df_copy["target"]
        df_copy = df_copy.drop("target", axis=1)
    else:
        raise ValueError("The DataFrame must contain a 'target' column.")

    # Initialize and fit scaler for features
    feature_scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        feature_scaler.fit_transform(df_copy),
        columns=df_copy.columns,
        index=df_copy.index,
    )

    # # Initialize and fit scaler for target
    # target_scaler = StandardScaler()
    scaled_target = target_col
    # scaled_target = pd.DataFrame(
    #     target_scaler.fit_transform(target_col.values.reshape(-1, 1)), columns=["target"], index=target_col.index
    # )
    # Save scaler parameters
    joblib.dump(feature_scaler, scaler_filename)
    # joblib.dump(target_scaler, target_scaler_filename)

    # Combine scaled features and target
    scaled_data = pd.concat([scaled_features, scaled_target], axis=1)

    # Restore datetime and ID columns if they existed
    if "measurement_time" in df.columns:
        scaled_data["measurement_time"] = datetime_col
    if "ID" in df.columns:
        scaled_data["ID"] = id_col

    return scaled_data


def transform_dataframe(df, scaler_filename):
    """
    Transform each column in the DataFrame using a pre-fitted StandardScaler
    loaded from a file.

    Args:
        df: pandas DataFrame to transform
        scaler_filename: file containing the pre-fitted scaler parameters

    Returns:
        Transformed DataFrame
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Separate datetime and ID columns if they exist
    if "measurement_time" in df_copy.columns:
        datetime_col = df_copy["measurement_time"]
        df_copy = df_copy.drop("measurement_time", axis=1)
    if "ID" in df_copy.columns:
        id_col = df_copy["ID"]
        df_copy = df_copy.drop("ID", axis=1)

    # Load the pre-fitted scaler and transform the data
    scaler = joblib.load(scaler_filename)
    transformed_data = pd.DataFrame(
        scaler.transform(df_copy), columns=df_copy.columns, index=df_copy.index
    )

    # Restore datetime and ID columns if they existed
    if "measurement_time" in df.columns:
        transformed_data["measurement_time"] = datetime_col
    if "ID" in df.columns:
        transformed_data["ID"] = id_col

    return transformed_data


def inverse_scale_dataframe(scaled_df, scaler_filename):
    """
    Perform inverse scaling on a scaled DataFrame using stored parameters.

    Args:
        scaled_df: scaled pandas DataFrame
        scaler_filename: file containing the scaler parameters

    Returns:
        Original-scale DataFrame
    """
    # Create a copy to avoid modifying the original
    df_copy = scaled_df.copy()

    # Separate datetime and ID columns if they exist
    if "measurement_time" in df_copy.columns:
        datetime_col = df_copy["measurement_time"]
        df_copy = df_copy.drop("measurement_time", axis=1)
    if "ID" in df_copy.columns:
        id_col = df_copy["ID"]
        df_copy = df_copy.drop("ID", axis=1)

    # Load scaler and inverse transform
    scaler = joblib.load(scaler_filename)
    original_data = pd.DataFrame(
        scaler.inverse_transform(df_copy), columns=df_copy.columns, index=df_copy.index
    )

    # Restore datetime and ID columns if they existed
    if "measurement_time" in scaled_df.columns:
        original_data["measurement_time"] = datetime_col
    if "ID" in scaled_df.columns:
        original_data["ID"] = id_col

    return original_data


def add_time_based_features(df):
    df = df.copy()
    df["measurement_time"] = pd.to_datetime(df["measurement_time"])

    # Extract hour, day of week, month
    df["hour"] = df["measurement_time"].dt.hour
    df["day_of_week"] = df["measurement_time"].dt.dayofweek  # Monday=0, Sunday=6
    df["month"] = df["measurement_time"].dt.month

    # Sine-cosine encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Is weekend
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Is holiday (placeholder, adjust according to your holiday calendar)
    holidays = []  # List of holiday dates in 'YYYY-MM-DD' format
    df["is_holiday"] = (
        df["measurement_time"]
        .dt.date.astype("datetime64[ns]")
        .isin(holidays)
        .astype(int)
    )

    # Season indicators
    def get_season(month):
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"

    df["season"] = df["month"].apply(get_season)

    # Define the categories for 'season' column
    season_categories = ["spring", "summer", "fall", "winter"]
    df["season"] = pd.Categorical(df["season"], categories=season_categories)

    # One-hot encode seasons with all categories
    df = pd.get_dummies(df, columns=["season"])

    return df


def add_temperature_difference_features(df):
    df = df.copy()

    # Calculate mean temperature from sources
    source_temps = [
        "source_1_temperature",
        "source_2_temperature",
        "source_3_temperature",
        "source_4_temperature",
    ]
    df["mean_source_temperature"] = df[source_temps].mean(axis=1)

    # Calculate differences between each source and the mean
    for col in source_temps:
        df[f"{col}_diff_mean"] = df[col] - df["mean_source_temperature"]

    # Calculate pairwise differences between sources
    for i in range(len(source_temps)):
        for j in range(i + 1, len(source_temps)):
            df[f"{source_temps[i]}_{source_temps[j]}_diff"] = (
                df[source_temps[i]] - df[source_temps[j]]
            )

    return df


def add_temperature_moving_averages(df):
    df = df.copy()
    df = df.sort_values("measurement_time")
    temp_cols = [
        "source_1_temperature",
        "source_2_temperature",
        "source_3_temperature",
        "source_4_temperature",
    ]

    for window in [1, 3, 24]:
        for col in temp_cols:
            df[f"{col}_ma_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )

    return df


def add_temperature_derivatives(df):
    df = df.copy()
    df = df.sort_values("measurement_time")
    temp_cols = [
        "source_1_temperature",
        "source_2_temperature",
        "source_3_temperature",
        "source_4_temperature",
    ]

    for col in temp_cols:
        df[f"{col}_rate_of_change"] = df[col].diff().fillna(0)

    return df


def add_temperature_lag_features(df):
    df = df.copy()
    df = df.sort_values("measurement_time")
    temp_cols = [
        "source_1_temperature",
        "source_2_temperature",
        "source_3_temperature",
        "source_4_temperature",
    ]

    for lag in [1, 3, 24]:
        for col in temp_cols:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag).fillna(method="bfill")

    return df


def add_solar_features(df):
    df = df.copy()
    sun_cols = [
        "sun_radiation_east",
        "sun_radiation_west",
        "sun_radiation_south",
        "sun_radiation_north",
        "sun_radiation_perpendicular",
    ]

    # Total and maximum solar radiation
    df["total_sun_radiation"] = df[sun_cols].sum(axis=1)
    df["max_sun_radiation"] = df[sun_cols].max(axis=1)

    # Solar radiation moving averages
    df = df.sort_values("measurement_time")
    for window in [1, 3, 24]:
        df[f"total_sun_radiation_ma_{window}h"] = (
            df["total_sun_radiation"].rolling(window=window, min_periods=1).mean()
        )

    return df


def add_wind_components(df):
    df = df.copy()
    # Convert wind direction from degrees to radians
    df["wind_direction_rad"] = np.deg2rad(df["wind_direction"])
    # Compute wind components
    df["wind_x"] = df["wind_speed"] * np.cos(df["wind_direction_rad"])
    df["wind_y"] = df["wind_speed"] * np.sin(df["wind_direction_rad"])
    return df


def add_wind_chill(df):
    df = df.copy()
    temp = df["outside_temperature"]
    wind_speed_kmh = df["wind_speed"] * 3.6  # Convert m/s to km/h
    wind_chill = (
        13.12
        + 0.6215 * temp
        - 11.37 * wind_speed_kmh**0.16
        + 0.3965 * temp * wind_speed_kmh**0.16
    )
    df["wind_chill"] = np.where((temp <= 10) & (wind_speed_kmh > 4.8), wind_chill, temp)
    return df


def add_fourier_terms(df, period, order, prefix):
    df = df.copy()
    timestamp = df["measurement_time"]
    t = (timestamp - timestamp.min()) / np.timedelta64(1, "h")  # Time in hours

    for i in range(1, order + 1):
        df[f"{prefix}_sin_{i}"] = np.sin(2 * np.pi * i * t / period)
        df[f"{prefix}_cos_{i}"] = np.cos(2 * np.pi * i * t / period)
    return df


def add_rolling_statistics(df, target_col):
    df = df.copy()
    df = df.sort_values("measurement_time")

    for window in [3, 6, 12, 24]:
        df[f"{target_col}_rolling_mean_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
        df[f"{target_col}_rolling_min_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).min()
        )
        df[f"{target_col}_rolling_max_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).max()
        )

    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        df[f"{target_col}_lag_{lag}h"] = (
            df[target_col].shift(lag).fillna(method="bfill")
        )

    return df


def add_autocorrelation_features(df, target_col):
    df = df.copy()
    df = df.sort_values("measurement_time")
    autocorr_values = []
    for i in range(len(df)):
        if i >= 24:
            autocorr = df[target_col].iloc[i - 24 : i].autocorr()
        else:
            autocorr = np.nan
        autocorr_values.append(autocorr)
    df[f"{target_col}_autocorr_24h"] = autocorr_values
    return df


def add_cross_correlation_features(df):
    df = df.copy()
    temp_cols = [
        "source_1_temperature",
        "source_2_temperature",
        "source_3_temperature",
        "source_4_temperature",
    ]
    df = df.sort_values("measurement_time")

    for i in range(len(temp_cols)):
        for j in range(i + 1, len(temp_cols)):
            corr_values = []
            for k in range(len(df)):
                if k >= 24:
                    corr = (
                        df[temp_cols[i]]
                        .iloc[k - 24 : k]
                        .corr(df[temp_cols[j]].iloc[k - 24 : k])
                    )
                else:
                    corr = np.nan
                corr_values.append(corr)
            df[f"{temp_cols[i]}_{temp_cols[j]}_corr_24h"] = corr_values

    return df


def add_temperature_management_features(df):
    df = df.copy()
    # Temperature differential
    df["temp_diff_outside_mean_room"] = (
        df["outside_temperature"] - df["mean_room_temperature"]
    )

    # HVAC load indicators (example: if outside temp is higher than a threshold and inside temp is below a threshold)
    df["hvac_cooling"] = (
        (df["outside_temperature"] > 25) & (df["mean_room_temperature"] < 22)
    ).astype(int)
    df["hvac_heating"] = (
        (df["outside_temperature"] < 15) & (df["mean_room_temperature"] > 22)
    ).astype(int)

    # Temperature comfort range violations
    df["temp_comfort_violation"] = (
        (df["mean_room_temperature"] < 20) | (df["mean_room_temperature"] > 24)
    ).astype(int)

    return df


def add_polynomial_interactions(df):
    df = df.copy()
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and ID columns if present
    numeric_cols = [col for col in numeric_cols if col not in ["target", "ID"]]
    inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interactions = inter.fit_transform(df[numeric_cols])
    interaction_cols = inter.get_feature_names_out(numeric_cols)
    interaction_df = pd.DataFrame(interactions, columns=interaction_cols)

    # Drop original columns to avoid duplication
    interaction_df = interaction_df.drop(columns=numeric_cols)
    df = pd.concat([df, interaction_df], axis=1)
    return df


def add_sine_cosine_transformations(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["target", "ID"]]
    new_columns = {f"{col}_sin": np.sin(df[col]) for col in numeric_cols}
    new_columns.update({f"{col}_cos": np.cos(df[col]) for col in numeric_cols})
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df


def add_all_interactions(df):
    df = df.copy()

    # Select numeric columns
    cols = [
        "source_1_temperature",
        "source_2_temperature",
        "source_3_temperature",
        "source_4_temperature",
        "mean_room_temperature",
        "sun_radiation_east",
        "sun_radiation_west",
        "sun_radiation_south",
        "sun_radiation_north",
        "sun_radiation_perpendicular",
        "outside_temperature",
        "wind_speed",
        "wind_direction",
        "clouds",
    ]

    # Add pairwise interactions
    for i, col1 in enumerate(cols):
        for col2 in cols[i + 1 :]:  # Avoid duplicate pairs
            # Add product feature
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    return df


def get_augmented_datasets(train_path, test_path):
    dataset_path = train_path
    submission_dataset_path = test_path
    timestamp_column = "measurement_time"

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

    submission_data = pd.read_csv(
        submission_dataset_path,
        parse_dates=[timestamp_column],
    )

    transformations = [
        fill_nan,
        add_time_based_features,
        add_temperature_difference_features,
        add_temperature_moving_averages,
        add_temperature_derivatives,
        add_temperature_lag_features,
        add_solar_features,
        add_wind_components,
        add_wind_chill,
        add_cross_correlation_features,
        lambda data: add_fourier_terms(data, period=24, order=3, prefix="daily"),
        lambda data: add_fourier_terms(data, period=168, order=3, prefix="weekly"),
        lambda data: add_fourier_terms(data, period=8760, order=3, prefix="yearly"),
        add_temperature_management_features,
        add_sine_cosine_transformations,
        add_all_interactions,
    ]

    # Apply transformations data
    for func in transformations:
        data = func(data)

    # Repeat for submission_data
    for func in transformations:
        submission_data = func(submission_data)

    y_train = data["target"].values
    X_train = data.drop(columns=["target", "ID", "measurement_time"])

    # prepare submission data for prediction
    training_columns = data.columns.drop("target")
    submission_data = submission_data.reindex(columns=training_columns, fill_value=0)
    test = submission_data.drop(columns=["ID", "measurement_time"])

    return X_train, y_train, test
