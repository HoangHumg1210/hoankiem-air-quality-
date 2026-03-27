


df = df.copy()

df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

for lag in [1, 3, 24, 72, 168]:
    df[f"PM25_lag_{lag}"] = df["PM25"].shift(lag)

shifted = df["PM25"].shift(1)
for window in [24, 72, 168]:
    df[f"PM25_roll_mean_{window}"] = shifted.rolling(window=window).mean()
    df[f"PM25_roll_std_{window}"] = shifted.rolling(window=window).std()
    df[f"PM25_roll_max_{window}"] = shifted.rolling(window=window).max()
    df[f"PM25_roll_min_{window}"] = shifted.rolling(window=window).min()

df["PM25_ewm_mean_24"] = shifted.ewm(span=24, adjust=False).mean()
df["PM25_ewm_mean_72"] = shifted.ewm(span=72, adjust=False).mean()
df["PM25_diff_1"] = shifted.diff(1)
df["PM25_diff_24"] = shifted.diff(24)

df = df.dropna().copy()

base_features = [
    "PM25_lag_1", "PM25_lag_3", "PM25_lag_24", "PM25_lag_72", "PM25_lag_168",
    "PM25_roll_mean_24", "PM25_roll_mean_72", "PM25_roll_mean_168",
    "PM25_roll_std_24", "PM25_roll_std_72", "PM25_roll_std_168",
    "PM25_roll_max_24", "PM25_roll_max_72", "PM25_roll_max_168",
    "PM25_roll_min_24", "PM25_roll_min_72", "PM25_roll_min_168",
    "PM25_ewm_mean_24", "PM25_ewm_mean_72",
    "PM25_diff_1", "PM25_diff_24",
]

weather_features = [
    "Temperature", "Pressure", "Wind Speed",
    "Clouds", "Precipitation", "Relative Humidity",
    "Accumulated Hours of Rain",
]

pollution_features = ["PM10", "CO", "NO2", "O3", "SO2"]

calendar_features = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "IsHoliday",
]

manual_v1_features = (
    base_features
    + ["PM10", "CO", "NO2"]
    + ["Temperature", "Pressure", "Wind Speed", "Relative Humidity", "Precipitation", "Clouds"]
    + calendar_features
)

production_v1_features = base_features + calendar_features

feature_groups = {
    "base": base_features,
    "weather": weather_features,
    "pollution": pollution_features,
    "calendar": calendar_features,
}

candidate_feature_sets = {
    "base": base_features,
    "base_weather": base_features + weather_features,
    "base_weather_pollution": base_features + weather_features + pollution_features,
    "manual_v1": manual_v1_features,
    "production_v1": production_v1_features,
    "optimistic_v1": manual_v1_features,
    "all": base_features + weather_features + pollution_features + calendar_features,
}

GRA_POOL_NAME = "all"
GRA_TOP_K = 12
GRA_RHO = 0.5


def _minmax_01(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def compute_gra_scores(X_df, y_series, rho=0.5):
    ref = _minmax_01(np.asarray(y_series, dtype=np.float64).reshape(-1))
    diffs = []
    normalized = {}

    for col in X_df.columns:
        seq = _minmax_01(X_df[col].to_numpy(dtype=np.float64))
        normalized[col] = seq
        diffs.append(np.abs(ref - seq))

    diff_matrix = np.vstack(diffs)
    delta_min = float(np.min(diff_matrix))
    delta_max = float(np.max(diff_matrix))
    if np.isclose(delta_max, 0.0):
        delta_max = 1.0

    rows = []
    for col in X_df.columns:
        diff = np.abs(ref - normalized[col])
        coeff = (delta_min + rho * delta_max) / (diff + rho * delta_max)
        rows.append({
            "feature": col,
            "gra_score": float(np.mean(coeff)),
        })

    return pd.DataFrame(rows).sort_values(["gra_score", "feature"], ascending=[False, True]).reset_index(drop=True)


for name, cols in candidate_feature_sets.items():
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        print(f"{name}: thieu cot -> {missing_cols}")
    else:
        print(f"{name}: {len(cols)} dac trung")

feature_cols = candidate_feature_sets["production_v1"]
print("Bo dac trung production mac dinh:", feature_cols)
""")
updates[17] = src("""
# ===== C?u hình GRU + Walk-forward =====
LOOKBACK = 336
CHUNK_HORIZON = 24
ROLLOUT_HORIZON = 72
N_CHUNKS = ROLLOUT_HORIZON // CHUNK_HORIZON
HORIZON = CHUNK_HORIZON
EVAL_SIZE = ROLLOUT_HORIZON
STEP_SIZE = ROLLOUT_HORIZON
MAX_FOLDS = 3
EPOCHS = 80
BATCH_SIZE = 32
SEED = 42
TARGET_TRANSFORM_MODE = "sqrt"
SELECTED_FEATURE_SET = "production_v1"
OPTIMISTIC_FEATURE_SET = "optimistic_v1"
INNER_VAL_SIZE = 336
PEAK_QUANTILE = 0.90
PEAK_WEIGHT = 4.0
HUBER_DELTA = 1.0
ROLLING_POLICY = "assimilated"
FEATURE_SET_COMPARE = [SELECTED_FEATURE_SET, OPTIMISTIC_FEATURE_SET]

np.random.seed(SEED)
tf.random.set_seed(SEED)

production_feature_cols = candidate_feature_sets[SELECTED_FEATURE_SET]
optimistic_feature_cols = candidate_feature_sets[OPTIMISTIC_FEATURE_SET]
feature_cols = production_feature_cols

print("B? d?c trung production dang dùng:", SELECTED_FEATURE_SET)
print("S? lu?ng d?c trung production:", len(production_feature_cols))
print("Danh sách c?t d?c trung production:", production_feature_cols)
print("B? d?i chi?u optimistic:", OPTIMISTIC_FEATURE_SET)
print("S? lu?ng d?c trung optimistic:", len(optimistic_feature_cols))
print("LOOKBACK / CHUNK_HORIZON / ROLLOUT_HORIZON:", LOOKBACK, CHUNK_HORIZON, ROLLOUT_HORIZON)
print("S? chunk rollout:", N_CHUNKS, "| step_size =", STEP_SIZE, "| max_folds =", MAX_FOLDS)
print("Rolling policy:", ROLLING_POLICY)
print("Ki?u bi?n d?i bi?n m?c tiêu:", TARGET_TRANSFORM_MODE)
print("Kích thu?c t?p validation n?i b?:", INNER_VAL_SIZE)
print("Hàm m?t mát có tr?ng s? cho d?nh: quantile =", PEAK_QUANTILE, "peak_weight =", PEAK_WEIGHT, "delta =", HUBER_DELTA)
print("C?u hình hu?n luy?n: EPOCHS =", EPOCHS, "BATCH_SIZE =", BATCH_SIZE)
""")

updates[19] = src("""
# ===== Sequence building + Seq2Seq GRU / Attention model =====

# T?o c?a s? tru?t cho bài toán multi-step forecasting.
def make_sequences(X, y, lookback=72, horizon=72):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    X_seq, y_seq = [], []
    max_start = len(X) - lookback - horizon + 1
    if max_start <= 0:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)

    for i in range(max_start):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback:i + lookback + horizon])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


# Xây d?ng mô hình Seq2Seq GRU v?i tùy ch?n attention và các tham s? c?u hình linh ho?t.
def build_gru_model(
    lookback,
    n_features,
    horizon,
    gru_units=(128, 64),
    dense_units=128,
    dropout=0.2,
    recurrent_dropout=0.0,
    learning_rate=5e-4,
    loss_fn="mse",
    l2_reg=0.0,
    clipnorm=1.0,
    use_attention=False,
):
    if isinstance(gru_units, int):
        gru_units = (gru_units, max(gru_units // 2, 32))
    elif len(gru_units) == 1:
        gru_units = (gru_units[0], max(gru_units[0] // 2, 32))

    encoder_units = gru_units[0]
    decoder_units = tuple(gru_units[1:]) if len(gru_units) > 1 else (max(encoder_units // 2, 32),)
    decoder_last_units = decoder_units[-1]
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None

    inputs = tf.keras.layers.Input(shape=(lookback, n_features))

    encoder_outputs = tf.keras.layers.GRU(
        encoder_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizer,
        name="encoder_gru",
    )(inputs)

    context = tf.keras.layers.Lambda(
        lambda t: t[:, -1, :],
        name="encoder_last_state",
    )(encoder_outputs)

    x = tf.keras.layers.RepeatVector(horizon, name="repeat_context")(context)

    for i, units in enumerate(decoder_units, start=1):
        x = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
            name=f"decoder_gru_{i}",
        )(x)

    decoder_outputs = x

    if use_attention:
        attention_values = encoder_outputs
        if encoder_units != decoder_last_units:
            attention_values = tf.keras.layers.Dense(
                decoder_last_units,
                kernel_regularizer=regularizer,
                name="encoder_attention_projection",
            )(attention_values)

        attention_context = tf.keras.layers.AdditiveAttention(name="temporal_attention")(
            [decoder_outputs, attention_values]
        )
        x = tf.keras.layers.Concatenate(name="decoder_attention_concat")(
            [decoder_outputs, attention_context]
        )
    else:
        x = decoder_outputs

    if dense_units:
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                dense_units,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
            ),
            name="time_distributed_dense",
        )(x)
        x = tf.keras.layers.Dropout(dropout, name="decoder_dropout")(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name="time_distributed_output")(x)
    outputs = tf.keras.layers.Reshape((horizon,), name="forecast_output")(x)
    model_name = "seq2seq_gru_attention" if use_attention else "seq2seq_gru"
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=loss_fn,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# T?o hàm m?t mát Huber có tr?ng s? cho các di?m d?nh (peak) d?a trên ngu?ng và tr?ng s? dã d?nh.
def make_weighted_huber_loss(peak_threshold, peak_weight=6.0, delta=1.0, horizon=72):
    peak_threshold = tf.constant(float(peak_threshold), dtype=tf.float32)
    peak_weight = tf.constant(float(peak_weight), dtype=tf.float32)
    delta = tf.constant(float(delta), dtype=tf.float32)
    step_weights = tf.reshape(tf.linspace(1.0, 1.8, horizon), (1, horizon))

    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        huber = tf.where(
            abs_error <= delta,
            0.5 * tf.square(error),
            delta * (abs_error - 0.5 * delta),
        )
        peak_weights = 1.0 + peak_weight * tf.cast(y_true >= peak_threshold, tf.float32)
        return tf.reduce_mean(huber * peak_weights * step_weights)

    return loss
""")
updates[22] = src("""
# ===== Data prep for train/eval =====

PRODUCTION_LAGS = [1, 3, 24, 72, 168]
PRODUCTION_WINDOWS = [24, 72, 168]


def build_history_feature_frame(raw_df):
    raw_df = raw_df.copy().sort_index()

    raw_df["hour"] = raw_df.index.hour
    raw_df["dayofweek"] = raw_df.index.dayofweek
    raw_df["month"] = raw_df.index.month
    raw_df["hour_sin"] = np.sin(2 * np.pi * raw_df["hour"] / 24)
    raw_df["hour_cos"] = np.cos(2 * np.pi * raw_df["hour"] / 24)
    raw_df["dow_sin"] = np.sin(2 * np.pi * raw_df["dayofweek"] / 7)
    raw_df["dow_cos"] = np.cos(2 * np.pi * raw_df["dayofweek"] / 7)
    raw_df["month_sin"] = np.sin(2 * np.pi * raw_df["month"] / 12)
    raw_df["month_cos"] = np.cos(2 * np.pi * raw_df["month"] / 12)

    for lag in PRODUCTION_LAGS:
        raw_df[f"PM25_lag_{lag}"] = raw_df["PM25"].shift(lag)

    shifted = raw_df["PM25"].shift(1)
    for window in PRODUCTION_WINDOWS:
        raw_df[f"PM25_roll_mean_{window}"] = shifted.rolling(window=window).mean()
        raw_df[f"PM25_roll_std_{window}"] = shifted.rolling(window=window).std()
        raw_df[f"PM25_roll_max_{window}"] = shifted.rolling(window=window).max()
        raw_df[f"PM25_roll_min_{window}"] = shifted.rolling(window=window).min()

    raw_df["PM25_ewm_mean_24"] = shifted.ewm(span=24, adjust=False).mean()
    raw_df["PM25_ewm_mean_72"] = shifted.ewm(span=72, adjust=False).mean()
    raw_df["PM25_diff_1"] = shifted.diff(1)
    raw_df["PM25_diff_24"] = shifted.diff(24)
    return raw_df


# Scale feature/target và t?o sequence cho hu?n luy?n và validation n?i b?.
def prepare_train_eval_sequences(train_X_df, train_y_df, eval_X_df, eval_y_df, lookback=72, horizon=72, target_mode="log1p"):
    if len(train_X_df) <= lookback:
        raise ValueError("train_X_df ph?i có ít nh?t `lookback` hàng")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(train_X_df.values)
    X_eval_scaled = x_scaler.transform(eval_X_df.values)

    y_train_scaled, y_scaler = transform_target(
        train_y_df.values.reshape(-1), scaler=None, fit=True, mode=target_mode
    )
    y_eval_scaled, _ = transform_target(
        eval_y_df.values.reshape(-1), scaler=y_scaler, fit=False, mode=target_mode
    )

    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, lookback=lookback, horizon=horizon)

    X_context = np.vstack([X_train_scaled[-lookback:], X_eval_scaled])
    y_context = np.concatenate([y_train_scaled[-lookback:], y_eval_scaled])
    X_eval_seq, y_eval_seq = make_sequences(X_context, y_context, lookback=lookback, horizon=horizon)

    return X_train_seq, y_train_seq, X_eval_seq, y_eval_seq, x_scaler, y_scaler


def build_inference_input(history_raw_df, feature_cols, x_scaler, lookback=72):
    feature_frame = build_history_feature_frame(history_raw_df)
    feature_frame = feature_frame.dropna(subset=feature_cols + ["PM25"]).copy()
    if len(feature_frame) < lookback:
        raise ValueError("Không d? l?ch s? sau khi t?o feature d? d?ng input inference.")

    X_window = feature_frame[feature_cols].tail(lookback).to_numpy(dtype=np.float32)
    X_scaled = x_scaler.transform(X_window)
    return X_scaled[np.newaxis, ...]


def summarize_rollout_predictions(rollout_df, peak_quantile=0.90):
    rollout_df = rollout_df.copy().sort_values("timestamp").reset_index(drop=True)
    rollout_df["pred_std"] = 0.0

    chunk_rows = []
    for chunk_id, chunk_df in rollout_df.groupby("chunk_id", sort=True):
        metrics = compute_regression_metrics(
            chunk_df["y_true"].to_numpy(),
            chunk_df["y_pred"].to_numpy(),
            peak_quantile=peak_quantile,
        )
        chunk_rows.append({
            "chunk_id": int(chunk_id),
            "n_points": int(len(chunk_df)),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "peak_mae": metrics["peak_mae"],
            "peak_threshold": metrics["peak_threshold"],
        })

    chunk_metrics_df = pd.DataFrame(chunk_rows)
    rollout_metrics = compute_regression_metrics(
        rollout_df["y_true"].to_numpy(),
        rollout_df["y_pred"].to_numpy(),
        peak_quantile=peak_quantile,
    )
    return {
        "timeline_df": rollout_df,
        "chunk_metrics_df": chunk_metrics_df,
        "rollout_metrics": rollout_metrics,
    }


def run_assimilated_rollout(
    model,
    history_raw_df,
    future_raw_df,
    feature_cols,
    x_scaler,
    y_scaler,
    lookback=72,
    chunk_horizon=24,
    rollout_horizon=72,
    target_mode="log1p",
):
    history_raw_df = history_raw_df.copy().sort_index()
    future_raw_df = future_raw_df.copy().sort_index().iloc[:rollout_horizon]
    if len(future_raw_df) < rollout_horizon:
        raise ValueError("Không d? d? li?u tuong lai d? rollout d? s? bu?c yêu c?u.")

    rows = []
    for chunk_start in range(0, rollout_horizon, chunk_horizon):
        chunk_id = chunk_start // chunk_horizon + 1
        chunk_future = future_raw_df.iloc[chunk_start:chunk_start + chunk_horizon].copy()
        if len(chunk_future) < chunk_horizon:
            raise ValueError("Chunk cu?i không d? s? bu?c d? báo.")

        X_input = build_inference_input(
            history_raw_df=history_raw_df,
            feature_cols=feature_cols,
            x_scaler=x_scaler,
            lookback=lookback,
        )
        y_pred_scaled = model.predict(X_input, verbose=0)
        y_pred = inverse_target(y_pred_scaled[0], y_scaler, mode=target_mode)

        for step_idx, (timestamp, y_true_value, y_pred_value) in enumerate(
            zip(chunk_future.index, chunk_future["PM25"].to_numpy(dtype=np.float64), y_pred),
            start=1,
        ):
            rows.append({
                "chunk_id": chunk_id,
                "chunk_step": step_idx,
                "global_step": chunk_start + step_idx,
                "timestamp": timestamp,
                "y_true": float(y_true_value),
                "y_pred": float(y_pred_value),
            })

        history_raw_df = pd.concat([history_raw_df, chunk_future], axis=0)

    rollout_df = pd.DataFrame(rows)
    if len(rollout_df) != rollout_horizon:
        raise ValueError("Rollout không tr? dúng s? bu?c yêu c?u.")

    return summarize_rollout_predictions(rollout_df, peak_quantile=PEAK_QUANTILE)
""")
cell23 = """
# ===== Walk-forward validate tren tap val, giu test hold-out =====

df_raw_rollout = df.copy().sort_index()
df_wf = build_history_feature_frame(df_raw_rollout).dropna().copy()

train_wf = df_wf[:train_end].copy()
val_wf = df_wf[val_start:val_end].copy()
test_wf = df_wf[test_start:].copy()

train_raw = df_raw_rollout[:train_end].copy()
val_raw = df_raw_rollout[val_start:val_end].copy()
test_raw = df_raw_rollout[test_start:].copy()

for feature_set_name in [SELECTED_FEATURE_SET, OPTIMISTIC_FEATURE_SET]:
    local_cols = candidate_feature_sets[feature_set_name]
    missing_feature_cols = [c for c in local_cols if c not in df_wf.columns]
    if missing_feature_cols:
        raise ValueError(f\"Thi?u các c?t feature cho {feature_set_name}: {missing_feature_cols}\")

MODEL_KWARGS = {
    \"gru_units\": (128, 64),
    \"dense_units\": 128,
    \"dropout\": 0.15,
    \"recurrent_dropout\": 0.0,
    \"learning_rate\": 7e-4,
    \"l2_reg\": 1e-5,
    \"clipnorm\": 1.0,
    \"use_attention\": True,
}
MODEL_LABEL = \"Seq2Seq GRU + Attention\" if MODEL_KWARGS.get(\"use_attention\", False) else \"Seq2Seq GRU\"


def make_callbacks():
    return [
        EarlyStopping(monitor=\"val_loss\", patience=10, restore_best_weights=True, min_delta=1e-4),
        ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=4, min_lr=1e-5),
    ]


def split_train_inner_val(train_X_df, train_y_df, inner_val_size):
    min_train_rows = LOOKBACK + CHUNK_HORIZON
    if len(train_X_df) <= inner_val_size + min_train_rows:
        raise ValueError(
            f\"Không d? d? li?u d? tách inner val. C?n > {inner_val_size + min_train_rows} rows, nh?n {len(train_X_df)}\"
        )
    train_core_X = train_X_df.iloc[:-inner_val_size].copy()
    train_core_y = train_y_df.iloc[:-inner_val_size].copy()
    inner_val_X = train_X_df.iloc[-inner_val_size:].copy()
    inner_val_y = train_y_df.iloc[-inner_val_size:].copy()
    return train_core_X, train_core_y, inner_val_X, inner_val_y


def fit_selector_model(train_core_X, train_core_y, inner_val_X, inner_val_y, model_kwargs=None, epochs=EPOCHS):
    model_kwargs = MODEL_KWARGS if model_kwargs is None else model_kwargs
    X_train_seq, y_train_seq, X_inner_val_seq, y_inner_val_seq, x_scaler, y_scaler = prepare_train_eval_sequences(
        train_core_X,
        train_core_y,
        inner_val_X,
        inner_val_y,
        lookback=LOOKBACK,
        horizon=CHUNK_HORIZON,
        target_mode=TARGET_TRANSFORM_MODE,
    )

    if len(X_train_seq) == 0 or len(X_inner_val_seq) == 0:
        raise ValueError(\"Không t?o du?c sequence cho train ho?c inner val. Ki?m tra l?i kích thu?c d? li?u và lookback/horizon.\")

    peak_threshold = float(np.quantile(y_train_seq.reshape(-1), PEAK_QUANTILE))
    loss_fn = make_weighted_huber_loss(
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT,
        delta=HUBER_DELTA,
        horizon=CHUNK_HORIZON,
    )

    model = build_gru_model(
        lookback=LOOKBACK,
        n_features=X_train_seq.shape[2],
        horizon=CHUNK_HORIZON,
        loss_fn=loss_fn,
        **model_kwargs,
    )
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_inner_val_seq, y_inner_val_seq),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=make_callbacks(),
        verbose=0,
    )
    best_epoch = int(np.argmin(history.history[\"val_loss\"])) + 1
    return model, history, best_epoch, x_scaler, y_scaler, peak_threshold


def fit_full_history_model(full_X_df, full_y_df, epochs, model_kwargs=None):
    model_kwargs = MODEL_KWARGS if model_kwargs is None else model_kwargs
    x_scaler = StandardScaler()
    X_full_scaled = x_scaler.fit_transform(full_X_df.values)
    y_full_scaled, y_scaler = transform_target(
        full_y_df.values.reshape(-1), scaler=None, fit=True, mode=TARGET_TRANSFORM_MODE
    )
    X_full_seq, y_full_seq = make_sequences(X_full_scaled, y_full_scaled, lookback=LOOKBACK, horizon=CHUNK_HORIZON)

    if len(X_full_seq) == 0:
        raise ValueError(\"Không t?o du?c sequence full-history\")

    peak_threshold = float(np.quantile(y_full_seq.reshape(-1), PEAK_QUANTILE))
    loss_fn = make_weighted_huber_loss(
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT,
        delta=HUBER_DELTA,
        horizon=CHUNK_HORIZON,
    )

    model = build_gru_model(
        lookback=LOOKBACK,
        n_features=X_full_seq.shape[2],
        horizon=CHUNK_HORIZON,
        loss_fn=loss_fn,
        **model_kwargs,
    )
    model.fit(
        X_full_seq,
        y_full_seq,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return model, x_scaler, y_scaler, peak_threshold
"""
cell23 += """

def fit_test_variant(feature_cols_local, history_feature_df, history_raw_df, future_raw_df, model_kwargs, selector_epochs=EPOCHS):
    history_X_df = history_feature_df[feature_cols_local].copy()
    history_y_df = history_feature_df[[\"PM25\"]].copy()

    train_core_X, train_core_y, inner_val_X, inner_val_y = split_train_inner_val(
        history_X_df, history_y_df, INNER_VAL_SIZE
    )
    selector_model, selector_history, best_epoch_local, _, _, peak_threshold_train = fit_selector_model(
        train_core_X,
        train_core_y,
        inner_val_X,
        inner_val_y,
        model_kwargs=model_kwargs,
        epochs=selector_epochs,
    )
    final_model, x_scaler_local, y_scaler_local, peak_threshold_full = fit_full_history_model(
        history_X_df,
        history_y_df,
        best_epoch_local,
        model_kwargs=model_kwargs,
    )
    rollout_summary = run_assimilated_rollout(
        model=final_model,
        history_raw_df=history_raw_df,
        future_raw_df=future_raw_df,
        feature_cols=feature_cols_local,
        x_scaler=x_scaler_local,
        y_scaler=y_scaler_local,
        lookback=LOOKBACK,
        chunk_horizon=CHUNK_HORIZON,
        rollout_horizon=ROLLOUT_HORIZON,
        target_mode=TARGET_TRANSFORM_MODE,
    )
    return {
        \"selector_model\": selector_model,
        \"selector_history\": selector_history,
        \"best_epoch\": best_epoch_local,
        \"train_peak_threshold_t\": peak_threshold_train,
        \"train_full_peak_threshold_t\": peak_threshold_full,
        \"rollout_summary\": rollout_summary,
    }


val_feature_pool = val_wf[feature_cols].copy()
base_train_X = train_wf[feature_cols].copy()
base_train_y = train_wf[[\"PM25\"]].copy()

fold_rows = []
compare_samples = []

for fold, start in enumerate(range(0, len(val_feature_pool) - ROLLOUT_HORIZON + 1, STEP_SIZE), start=1):
    if fold > MAX_FOLDS:
        break

    end = start + ROLLOUT_HORIZON
    fold_eval_raw = val_raw.iloc[start:end].copy()
    if len(fold_eval_raw) < ROLLOUT_HORIZON:
        break

    fold_history_X = pd.concat([base_train_X, val_feature_pool.iloc[:start]], axis=0)
    fold_history_y = pd.concat([base_train_y, val_wf.iloc[:start][[\"PM25\"]]], axis=0)
    train_core_X, train_core_y, inner_val_X, inner_val_y = split_train_inner_val(
        fold_history_X, fold_history_y, INNER_VAL_SIZE
    )

    model, history, best_epoch, x_scaler, y_scaler, peak_threshold_train = fit_selector_model(
        train_core_X, train_core_y, inner_val_X, inner_val_y
    )

    fold_history_raw = pd.concat([train_raw, val_raw.iloc[:start]], axis=0)
    rollout_summary = run_assimilated_rollout(
        model=model,
        history_raw_df=fold_history_raw,
        future_raw_df=fold_eval_raw,
        feature_cols=feature_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        lookback=LOOKBACK,
        chunk_horizon=CHUNK_HORIZON,
        rollout_horizon=ROLLOUT_HORIZON,
        target_mode=TARGET_TRANSFORM_MODE,
    )

    chunk_metrics_df = rollout_summary[\"chunk_metrics_df\"].set_index(\"chunk_id\")
    rollout_metrics = rollout_summary[\"rollout_metrics\"]
    fold_record = {
        \"fold\": fold,
        \"train_rows\": len(fold_history_X),
        \"inner_val_rows\": len(inner_val_X),
        \"rollout_rows\": len(fold_eval_raw),
        \"rollout_mae\": rollout_metrics[\"mae\"],
        \"rollout_rmse\": rollout_metrics[\"rmse\"],
        \"rollout_mape\": rollout_metrics[\"mape\"],
        \"rollout_peak_mae\": rollout_metrics[\"peak_mae\"],
        \"rollout_peak_threshold\": rollout_metrics[\"peak_threshold\"],
        \"train_peak_threshold_t\": peak_threshold_train,
        \"best_epoch\": best_epoch,
        \"best_inner_val_loss\": float(np.min(history.history[\"val_loss\"])),
    }

    for chunk_id in range(1, N_CHUNKS + 1):
        chunk_metrics = chunk_metrics_df.loc[chunk_id]
        fold_record[f\"chunk_{chunk_id}_mae\"] = chunk_metrics[\"mae\"]
        fold_record[f\"chunk_{chunk_id}_rmse\"] = chunk_metrics[\"rmse\"]
        fold_record[f\"chunk_{chunk_id}_mape\"] = chunk_metrics[\"mape\"]
        fold_record[f\"chunk_{chunk_id}_peak_mae\"] = chunk_metrics[\"peak_mae\"]

    compare_samples.append({
        \"fold\": fold,
        \"timeline_df\": rollout_summary[\"timeline_df\"].copy(),
        \"chunk_metrics_df\": rollout_summary[\"chunk_metrics_df\"].copy(),
    })
    fold_rows.append(fold_record)

walkforward_df = pd.DataFrame(fold_rows)
print(\"=== Walk-forward validation: production 72h rollout = 24h x 3 ===\")
display(walkforward_df)

if not walkforward_df.empty:
    val_summary_df = pd.DataFrame([
        {
            \"mean_rollout_mae\": walkforward_df[\"rollout_mae\"].mean(),
            \"mean_rollout_rmse\": walkforward_df[\"rollout_rmse\"].mean(),
            \"mean_chunk_1_mae\": walkforward_df[\"chunk_1_mae\"].mean(),
            \"mean_chunk_2_mae\": walkforward_df[\"chunk_2_mae\"].mean(),
            \"mean_chunk_3_mae\": walkforward_df[\"chunk_3_mae\"].mean(),
        }
    ])
    print(\"\\n=== Validation summary ===\")
    display(val_summary_df)


# ===== Khoa config tren validation, train tren train+val va danh gia 1 lan tren test =====
train_val_feature_df = pd.concat([train_wf, val_wf], axis=0)
train_val_raw = pd.concat([train_raw, val_raw], axis=0)
test_rollout_raw = test_raw.iloc[:ROLLOUT_HORIZON].copy()

if len(test_rollout_raw) < ROLLOUT_HORIZON:
    raise ValueError(\"Không d? d? li?u test d? rollout d? 72h.\")

test_variant = fit_test_variant(
    feature_cols_local=feature_cols,
    history_feature_df=train_val_feature_df,
    history_raw_df=train_val_raw,
    future_raw_df=test_rollout_raw,
    model_kwargs=MODEL_KWARGS,
    selector_epochs=EPOCHS,
)

best_epoch_test = test_variant[\"best_epoch\"]
peak_threshold_test_train = test_variant[\"train_peak_threshold_t\"]
peak_threshold_test_full = test_variant[\"train_full_peak_threshold_t\"]
test_eval_summary = test_variant[\"rollout_summary\"]
test_timeline_df = test_eval_summary[\"timeline_df\"].copy()
test_chunk_metrics_df = test_eval_summary[\"chunk_metrics_df\"].copy()
test_rollout_metrics = test_eval_summary[\"rollout_metrics\"]

test_metrics_df = pd.DataFrame([
    {
        \"best_epoch\": best_epoch_test,
        \"test_rollout_mae\": test_rollout_metrics[\"mae\"],
        \"test_rollout_rmse\": test_rollout_metrics[\"rmse\"],
        \"test_rollout_mape\": test_rollout_metrics[\"mape\"],
        \"test_rollout_peak_mae\": test_rollout_metrics[\"peak_mae\"],
        \"train_peak_threshold_t\": peak_threshold_test_train,
        \"train_val_peak_threshold_t\": peak_threshold_test_full,
    }
])
"""
cell23 += """
test_report_df = pd.DataFrame([
    {
        \"model\": MODEL_LABEL,
        \"rollout_horizon\": ROLLOUT_HORIZON,
        \"chunk_horizon\": CHUNK_HORIZON,
        \"mae\": test_rollout_metrics[\"mae\"],
        \"rmse\": test_rollout_metrics[\"rmse\"],
        \"MAPE\": test_rollout_metrics[\"mape\"],
        \"best_epoch\": best_epoch_test,
    }
])

print(\"\\n=== Hold-out test rollout metrics ===\")
display(test_metrics_df)
print(\"\\n=== Chunk-level test metrics ===\")
display(test_chunk_metrics_df)
print(\"\\n=== Presentation table ===\")
display(test_report_df)

plot_df = test_timeline_df.copy()

plt.figure(figsize=(15, 5))
plt.plot(plot_df[\"timestamp\"], plot_df[\"y_true\"], label=\"Th?c t? (Test)\", linewidth=1.8, color=\"#4C72B0\")
plt.plot(plot_df[\"timestamp\"], plot_df[\"y_pred\"], label=MODEL_LABEL, linewidth=1.8, color=\"#DD8452\")

for chunk_boundary in range(CHUNK_HORIZON, ROLLOUT_HORIZON, CHUNK_HORIZON):
    boundary_ts = plot_df.iloc[chunk_boundary][\"timestamp\"]
    plt.axvline(boundary_ts, color=\"#999999\", linestyle=\"--\", linewidth=1.0, alpha=0.7)

plt.title(\"Production-style 72h rollout tren test (24h x 3 assimilated)\")
plt.xlabel(\"\")
plt.ylabel(\"PM2.5\")
plt.legend()
plt.grid(alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
"""
updates[23] = src(cell23)
updates[24] = src("""
# ===== Feature-set comparison: production vs optimistic =====

FEATURE_COMPARE_MODEL_LABEL = "Seq2Seq GRU + Attention"
FEATURE_COMPARE_MODEL_KWARGS = {**MODEL_KWARGS, "use_attention": True}
FEATURE_COMPARE_MAX_EPOCHS = min(EPOCHS, 30)


# Ch?y cùng m?t ki?n trúc model trên 2 b? feature d? th?y chênh l?ch gi?a production và optimistic.
def run_feature_set_variant(feature_set_name, model_label, model_kwargs):
    local_feature_cols = candidate_feature_sets[feature_set_name]
    local_variant = fit_test_variant(
        feature_cols_local=local_feature_cols,
        history_feature_df=train_val_feature_df,
        history_raw_df=train_val_raw,
        future_raw_df=test_rollout_raw,
        model_kwargs=model_kwargs,
        selector_epochs=FEATURE_COMPARE_MAX_EPOCHS,
    )
    rollout_summary = local_variant["rollout_summary"]
    rollout_metrics = rollout_summary["rollout_metrics"]

    return {
        "feature_set": feature_set_name,
        "label": model_label,
        "best_epoch": local_variant["best_epoch"],
        "n_features": len(local_feature_cols),
        "metrics": {
            "feature_set": feature_set_name,
            "n_features": len(local_feature_cols),
            "best_epoch": local_variant["best_epoch"],
            "test_rollout_mae": rollout_metrics["mae"],
            "test_rollout_rmse": rollout_metrics["rmse"],
            "test_rollout_mape": rollout_metrics["mape"],
            "test_rollout_peak_mae": rollout_metrics["peak_mae"],
        },
        "chunk_metrics_df": rollout_summary["chunk_metrics_df"].copy(),
        "timeline_df": rollout_summary["timeline_df"].copy(),
    }


feature_compare_results = []
for feature_set_name in FEATURE_SET_COMPARE:
    print(f"Dang danh gia feature set: {feature_set_name}")
    feature_compare_results.append(
        run_feature_set_variant(
            feature_set_name=feature_set_name,
            model_label=FEATURE_COMPARE_MODEL_LABEL,
            model_kwargs=FEATURE_COMPARE_MODEL_KWARGS,
        )
    )

feature_compare_df = pd.DataFrame([res["metrics"] for res in feature_compare_results])
feature_compare_df = feature_compare_df.sort_values(["test_rollout_mae", "test_rollout_rmse"]).reset_index(drop=True)
production_rollout_mae = float(
    feature_compare_df.loc[feature_compare_df["feature_set"] == SELECTED_FEATURE_SET, "test_rollout_mae"].iloc[0]
)
feature_compare_df["delta_vs_production_mae"] = feature_compare_df["test_rollout_mae"] - production_rollout_mae
print("=== So sanh feature set tren test rollout 72h ===")
display(feature_compare_df)

best_feature_set = feature_compare_df.loc[0, "feature_set"]
print("Feature set co MAE rollout tot nhat:", best_feature_set)
if best_feature_set != SELECTED_FEATURE_SET:
    print("Luu y: optimistic_v1 tot hon tren test rollout cho thay nguy co chenh lech giua bo feature nghien cuu va bo feature production.")
""")

updates[25] = src("""
# ===== Presentation dashboard: Seq2Seq GRU vs Seq2Seq GRU + Attention =====

COMPARE_MODEL_CONFIGS = [
    ("Seq2Seq GRU", {**MODEL_KWARGS, "use_attention": False}),
    ("Seq2Seq GRU + Attention", {**MODEL_KWARGS, "use_attention": True}),
]
COMPARE_REUSE_CURRENT_RESULTS = True
COMPARE_MAX_EPOCHS = min(EPOCHS, 30)


# Ðóng gói k?t qu? dã có s?n t? cell test chính d? tránh train l?i m?t mô hình.
def build_cached_result(model_label, model_kwargs):
    return {
        "label": model_label,
        "model_kwargs": model_kwargs,
        "best_epoch": best_epoch_test,
        "metrics": {
            "model": model_label,
            "mae": test_rollout_metrics["mae"],
            "rmse": test_rollout_metrics["rmse"],
            "MAPE": test_rollout_metrics["mape"],
        },
        "timeline_df": test_timeline_df.copy(),
    }


# Hu?n luy?n và dánh giá m?t bi?n th? model trên production pipeline.
def run_test_variant(model_label, model_kwargs):
    variant = fit_test_variant(
        feature_cols_local=feature_cols,
        history_feature_df=train_val_feature_df,
        history_raw_df=train_val_raw,
        future_raw_df=test_rollout_raw,
        model_kwargs=model_kwargs,
        selector_epochs=COMPARE_MAX_EPOCHS,
    )
    rollout_summary = variant["rollout_summary"]
    rollout_metrics = rollout_summary["rollout_metrics"]

    return {
        "label": model_label,
        "model_kwargs": model_kwargs,
        "best_epoch": variant["best_epoch"],
        "metrics": {
            "model": model_label,
            "mae": rollout_metrics["mae"],
            "rmse": rollout_metrics["rmse"],
            "MAPE": rollout_metrics["mape"],
        },
        "timeline_df": rollout_summary["timeline_df"].copy(),
    }


presentation_results = []
for model_label, model_kwargs in COMPARE_MODEL_CONFIGS:
    can_reuse_current = (
        COMPARE_REUSE_CURRENT_RESULTS
        and "test_timeline_df" in globals()
        and model_label == MODEL_LABEL
    )

    if can_reuse_current:
        print(f"Tái sử dụng kết quả dã có cho: {model_label}")
        presentation_results.append(build_cached_result(model_label, model_kwargs))
    else:
        print(f"Đang train để so sánh cho: {model_label}")
        presentation_results.append(run_test_variant(model_label, model_kwargs))

presentation_metrics_df = pd.DataFrame([res["metrics"] for res in presentation_results])
presentation_metrics_df = presentation_metrics_df.sort_values("mae").reset_index(drop=True)
print("=== Bảng so sánh ===")
display(presentation_metrics_df)

seq2seq_res = next(res for res in presentation_results if res["label"] == "Seq2Seq GRU")
attn_res = next(res for res in presentation_results if res["label"] == "Seq2Seq GRU + Attention")

presentation_df = seq2seq_res["timeline_df"][["timestamp", "y_true"]].copy()
presentation_df = presentation_df.rename(columns={"y_true": "actual"})
presentation_df["seq2seq_gru"] = seq2seq_res["timeline_df"]["y_pred"].to_numpy()
presentation_df["seq2seq_gru_attention"] = attn_res["timeline_df"]["y_pred"].to_numpy()

plt.figure(figsize=(16, 6))
plt.plot(presentation_df["timestamp"], presentation_df["actual"], label="Thực tế (Test)", linewidth=1.9, color="tab:blue")
plt.plot(presentation_df["timestamp"], presentation_df["seq2seq_gru"], label="Seq2Seq GRU", linewidth=1.6, alpha=0.95, color="tab:green")
plt.plot(presentation_df["timestamp"], presentation_df["seq2seq_gru_attention"], label="Seq2Seq GRU + Attention", linewidth=1.6, alpha=0.95, color="tab:red")
for chunk_boundary in range(CHUNK_HORIZON, ROLLOUT_HORIZON, CHUNK_HORIZON):
    boundary_ts = presentation_df.iloc[chunk_boundary]["timestamp"]
    plt.axvline(boundary_ts, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
plt.title("Production-style 72h rollout trên test")
plt.xlabel("Thời gian")
plt.ylabel("PM2.5")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

zoom_hours = ROLLOUT_HORIZON
zoom_df = presentation_df.tail(zoom_hours).copy()
plt.figure(figsize=(16, 5))
plt.plot(zoom_df["timestamp"], zoom_df["actual"], label="Thực tế (Test)", linewidth=1.9, color="tab:blue")
plt.plot(zoom_df["timestamp"], zoom_df["seq2seq_gru"], label="Seq2Seq GRU", linewidth=1.6, alpha=0.95, color="tab:green")
plt.plot(zoom_df["timestamp"], zoom_df["seq2seq_gru_attention"], label="Seq2Seq GRU + Attention", linewidth=1.6, alpha=0.95, color="tab:red")
for chunk_boundary in range(CHUNK_HORIZON, ROLLOUT_HORIZON, CHUNK_HORIZON):
    boundary_ts = zoom_df.iloc[chunk_boundary]["timestamp"]
    plt.axvline(boundary_ts, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
plt.title("Zoom toàn bộ rollout 72h trên test")
plt.xlabel("Thời gian")
plt.ylabel("PM2.5")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



