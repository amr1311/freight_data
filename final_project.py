
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
METRICS_PATH = ROOT / "monthly_validation_metrics.csv"
MODEL_PATH = ROOT / "catboost_freight_rate_model.cbm"

TARGET = "posted_rate"

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

DATE_FEATURES = [
    "month",
    "day_of_week",
    "day_of_month",
]

MODEL_PARAMS = {
    "iterations": 1200,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 15,
    "random_strength": 5.0,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


# ============================================================
# 2. Helper functions
# ============================================================

def add_date_features(data):
    """
    Extract date-based features without using the raw date
    as a CatBoost feature.
    """
    data = data.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day

    return data


def remove_iqr_outliers(data, numeric_columns):
    """
    Remove rows containing outliers according to the IQR rule.

    Note:
    This applies to the supplied numeric columns, including
    the target if TARGET is included in numeric_columns.
    """
    data = data.copy()
    keep_rows = pd.Series(True, index=data.index)

    for column in numeric_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        valid_values = values.dropna()

        if valid_values.empty:
            continue

        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        column_mask = (
            values.isna()
            | values.between(lower_bound, upper_bound)
        )

        keep_rows &= column_mask

    cleaned_data = data.loc[keep_rows].copy()

    removed_count = len(data) - len(cleaned_data)
    removed_percentage = (
        removed_count / len(data) * 100
        if len(data) > 0
        else 0
    )

    print("\nOutlier removal")
    print("----------------")
    print(f"Original rows: {len(data):,}")
    print(f"Removed rows: {removed_count:,}")
    print(f"Removed percentage: {removed_percentage:.2f}%")
    print(f"Remaining rows: {len(cleaned_data):,}")

    return cleaned_data


def prepare_model_features(
    data,
    features,
    categorical_features,
    numeric_medians,
    numeric_bounds=None,
):
    """
    Prepare a dataset using the same feature columns and
    missing-value handling for training and prediction.
    """
    data = data.copy()

    # Create date features before selecting model columns.
    if "date" in data.columns:
        data = add_date_features(data)

    # Add any missing model columns, such as market_index
    # or quote_signal, when they are absent from prediction data.
    data = data.reindex(
        columns=list(dict.fromkeys(
            list(data.columns) + list(features)
        ))
    )

    X = data.reindex(columns=features).copy()

    # Clean categorical features.
    for column in categorical_features:
        if column in X.columns:
            X[column] = (
                X[column]
                .fillna("Unknown")
                .astype(str)
            )

    # Clean numeric features and fill missing values.
    numeric_features = [
        column for column in features
        if column not in categorical_features
    ]

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        median_value = numeric_medians.get(column, 0.0)

        X[column] = X[column].fillna(median_value)

        # Clip prediction values using bounds learned from
        # the training dataset, when bounds are supplied.
        if numeric_bounds and column in numeric_bounds:
            lower, upper = numeric_bounds[column]
            X[column] = X[column].clip(
                lower=lower,
                upper=upper
            )

    return X


def calculate_numeric_medians(data, numeric_features):
    """Calculate medians using only the supplied training data."""
    medians = {}

    for column in numeric_features:
        values = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        median_value = values.median()

        if pd.isna(median_value):
            median_value = 0.0

        medians[column] = float(median_value)

    return medians


def calculate_iqr_bounds(data, numeric_features):
    """Calculate feature clipping bounds from training data."""
    bounds = {}

    for column in numeric_features:
        values = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        bounds[column] = (
            float(q1 - 1.5 * iqr),
            float(q3 + 1.5 * iqr),
        )

    return bounds


def create_model():
    """Create a new CatBoost regression model."""
    return CatBoostRegressor(**MODEL_PARAMS)


# ============================================================
# 3. Load and clean the training data
# ============================================================

print("=" * 65)
print("FREIGHT RATE PREDICTION")
print("=" * 65)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_PATH}"
    )

df = pd.read_csv(TRAIN_PATH)

print(f"\nOriginal dataset shape: {df.shape}")
print("\nOriginal columns:")
print(df.columns.tolist())

required_columns = [
    "date",
    TARGET,
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required training columns: {missing_columns}"
    )

# Remove rows with missing target values.
df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(subset=[TARGET]).copy()

# Parse dates and remove rows with invalid dates.
df = add_date_features(df)
df = df.dropna(subset=["date"]).copy()

# Sort chronologically for time-based validation.
df = df.sort_values("date").reset_index(drop=True)

# Create a month identifier for validation splitting only.
df["year_month"] = df["date"].dt.to_period("M")

# Identify numeric columns for outlier detection.
numeric_columns_for_outliers = (
    df.select_dtypes(include=[np.number])
    .columns.tolist()
)

# Remove outliers using the IQR rule.
df = remove_iqr_outliers(
    df,
    numeric_columns_for_outliers
)

df = df.sort_values("date").reset_index(drop=True)

if df.empty:
    raise ValueError(
        "No training rows remain after data cleaning."
    )

# Fill missing numeric values using medians.
numeric_columns = (
    df.select_dtypes(include=[np.number])
    .columns.tolist()
)

for column in numeric_columns:
    median_value = df[column].median()

    if pd.isna(median_value):
        median_value = 0.0

    df[column] = df[column].fillna(median_value)

# Fill missing categorical values.
for column in CATEGORICAL_FEATURES:
    if column in df.columns:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

# Keep a chronological month identifier.
df["year_month"] = df["date"].dt.to_period("M")

print(f"\nCleaned dataset shape: {df.shape}")
print(f"First date: {df['date'].min()}")
print(f"Last date:  {df['date'].max()}")


# ============================================================
# 4. Define features and preprocessing information
# ============================================================

EXCLUDED_COLUMNS = {
    TARGET,
    "date",
    "year_month",
    "load_id",
    "predicted_rate",
}

features = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

categorical_features = [
    column
    for column in CATEGORICAL_FEATURES
    if column in features
]

numeric_features = [
    column
    for column in features
    if column not in categorical_features
]

print("\nModel features:")
for index, feature in enumerate(features, start=1):
    print(f"{index}. {feature}")

print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)

# These statistics are used for missing-value handling
# and prediction-time clipping.
numeric_medians = calculate_numeric_medians(
    df,
    numeric_features
)

numeric_bounds = calculate_iqr_bounds(
    df,
    numeric_features
)


# ============================================================
# 5. Monthly expanding-window validation
# ============================================================

print("\n" + "=" * 65)
print("MONTHLY EXPANDING-WINDOW VALIDATION")
print("=" * 65)

months = sorted(df["year_month"].unique())
monthly_results = []

if len(months) < 2:
    raise ValueError(
        "At least two different months are required "
        "for monthly expanding-window validation."
    )

for index in range(1, len(months)):
    train_months = months[:index]
    validation_month = months[index]

    train_data = df[
        df["year_month"].isin(train_months)
    ].copy()

    validation_data = df[
        df["year_month"] == validation_month
    ].copy()

    if train_data.empty or validation_data.empty:
        continue

    X_train = prepare_model_features(
        train_data,
        features,
        categorical_features,
        numeric_medians,
        numeric_bounds
    )

    X_valid = prepare_model_features(
        validation_data,
        features,
        categorical_features,
        numeric_medians,
        numeric_bounds
    )

    y_train = train_data[TARGET].astype(float)
    y_valid = validation_data[TARGET].astype(float)

    model = create_model()

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features
    )

    predictions = model.predict(X_valid)

    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    monthly_results.append({
        "train_start_month": str(train_months[0]),
        "train_end_month": str(train_months[-1]),
        "validation_month": str(validation_month),
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "RMSE": rmse,
        "MAE": mae,
    })

    print(
        f"Train: {train_months[0]} to {train_months[-1]} "
        f"| Validate: {validation_month} "
        f"| RMSE: {rmse:.2f} "
        f"| MAE: {mae:.2f}"
    )

metrics_df = pd.DataFrame(monthly_results)
metrics_df.to_csv(METRICS_PATH, index=False)

if not metrics_df.empty:
    print("\nValidation summary")
    print("------------------")
    print(f"Average monthly RMSE: {metrics_df['RMSE'].mean():.2f}")
    print(f"Average monthly MAE:  {metrics_df['MAE'].mean():.2f}")
    print(f"\nMetrics saved to: {METRICS_PATH}")
else:
    print("No monthly validation results were generated.")


# ============================================================
# 6. Train final model on all cleaned training data
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINAL MODEL")
print("=" * 65)

X_all = prepare_model_features(
    df,
    features,
    categorical_features,
    numeric_medians,
    numeric_bounds
)

y_all = df[TARGET].astype(float)

final_model = create_model()

final_model.fit(
    X_all,
    y_all,
    cat_features=categorical_features
)

final_model.save_model(str(MODEL_PATH))

print(f"Final model saved to: {MODEL_PATH}")


# ============================================================
# 7. Predict validation.csv
# ============================================================

print("\n" + "=" * 65)
print("PREDICTING VALIDATION DATA")
print("=" * 65)

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_PATH}"
    )

validation = pd.read_csv(VALIDATION_PATH)

if "load_id" not in validation.columns:
    raise ValueError(
        "validation.csv must contain a load_id column."
    )

validation = add_date_features(validation)

X_validation = prepare_model_features(
    validation,
    features,
    categorical_features,
    numeric_medians,
    numeric_bounds
)

validation_predictions = final_model.predict(X_validation)

if not np.isfinite(validation_predictions).all():
    raise ValueError(
        "Validation predictions contain NaN or infinite values."
    )

if (validation_predictions <= 0).any():
    raise ValueError(
        "Validation predictions contain non-positive rates."
    )

prediction_results = pd.DataFrame({
    "load_id": validation["load_id"].values,
    "predicted_rate": validation_predictions,
})

# If a template exists, preserve its load order and IDs.
if TEMPLATE_PATH.exists():
    template = pd.read_csv(TEMPLATE_PATH)

    if "load_id" not in template.columns:
        raise ValueError(
            "The prediction template must contain load_id."
        )

    if template["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values found in the prediction template."
        )

    if prediction_results["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values found in validation predictions."
        )

    final_predictions = template[["load_id"]].merge(
        prediction_results,
        on="load_id",
        how="left",
        validate="one_to_one"
    )

    if final_predictions["predicted_rate"].isna().any():
        missing_count = final_predictions[
            "predicted_rate"
        ].isna().sum()

        raise ValueError(
            f"{missing_count} template IDs have no matching prediction."
        )

    if len(final_predictions) != len(template):
        raise ValueError(
            "Final prediction row count does not match the template."
        )
else:
    final_predictions = prediction_results

final_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print(f"Validation rows: {len(final_predictions):,}")
print(
    "Missing predictions: "
    f"{final_predictions['predicted_rate'].isna().sum()}"
)
print(f"Saved to: {PREDICTIONS_PATH}")

print("\nFirst five predictions:")
print(final_predictions.head())


# ============================================================
# 8. Predict the December chart inputs
# ============================================================

print("\n" + "=" * 65)
print("PREDICTING DECEMBER CHART INPUTS")
print("=" * 65)

if not DECEMBER_PATH.exists():
    raise FileNotFoundError(
        f"December input file not found: {DECEMBER_PATH}"
    )

december = pd.read_csv(DECEMBER_PATH)

required_december_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
]

missing_december_columns = [
    column
    for column in required_december_columns
    if column not in december.columns
]

if missing_december_columns:
    raise ValueError(
        "Missing required December columns: "
        f"{missing_december_columns}"
    )

# Extract the same date features used during training.
december = add_date_features(december)

X_december = prepare_model_features(
    december,
    features,
    categorical_features,
    numeric_medians,
    numeric_bounds
)

december_predictions = final_model.predict(X_december)

if not np.isfinite(december_predictions).all():
    raise ValueError(
        "December predictions contain NaN or infinite values."
    )

if (december_predictions <= 0).any():
    raise ValueError(
        "December predictions contain non-positive rates."
    )

december["predicted_rate"] = december_predictions

# Keep the required output columns and original row order.
december_output = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
].copy()

december_output.to_csv(
    DECEMBER_PATH,
    index=False
)

print(f"December rows: {len(december_output)}")
print(
    "Unique December predictions: "
    f"{december_output['predicted_rate'].nunique()}"
)
print(
    "Prediction minimum: "
    f"{december_output['predicted_rate'].min():.2f}"
)
print(
    "Prediction maximum: "
    f"{december_output['predicted_rate'].max():.2f}"
)
print(f"Updated file: {DECEMBER_PATH}")

print("\nDecember predictions:")
print(
    december_output[
        ["date", "predicted_rate"]
    ].to_string(index=False)
)


# ============================================================
# 9. Completion summary
# ============================================================

print("\n" + "=" * 65)
print("PROJECT RUN COMPLETED")
print("=" * 65)

print(f"1. Validation predictions: {PREDICTIONS_PATH}")
print(f"2. December predictions:   {DECEMBER_PATH}")
print(f"3. Monthly metrics:        {METRICS_PATH}")
print(f"4. Saved CatBoost model:   {MODEL_PATH}")

print("\nNext step:")
print(
    "Run the company's original scorer using the command "
    "shown below."
)
