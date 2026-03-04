import os
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features
from src.model_training import train_model


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = load_data("data/raw/cardio_data.csv")
    df = clean_data(df)
    df = create_features(df)

    df.to_csv("data/processed/processed_data.csv", index=False)

    train_model(df)


if __name__ == "__main__":
    main()