def create_features(df):

    # BMI
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Pulse Pressure
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    return df