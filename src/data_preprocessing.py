import pandas as pd

def load_data(path):
    # IMPORTANT: sep=";"
    df = pd.read_csv(path, sep=";")
    return df

def clean_data(df):

    df = df.drop_duplicates()

    # Convert age from days to years
    df["age"] = df["age"] / 365

    # Encode gender (1=women, 2=men → convert to 0/1)
    df["gender"] = df["gender"].map({1: 0, 2: 1})

    # Remove unrealistic BP values
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 150)]

    df = df.fillna(df.median(numeric_only=True))

    return df