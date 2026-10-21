from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = Path("ML_fretting_data.csv")

try:
    df = pd.read_csv(path, encoding="cp1252")
    print("Dataset loaded successfully.")
    print("Columns of the DataFrame:", df.columns.tolist())
    print("Shape of the dataset:", df.shape)

    print("\nDataset information:")
    df.info()

    print("\nMissing values in original dataset:")
    print(df.isnull().sum())

    # Dropping rows with null values
    df_ml = df.dropna().reset_index(drop=True)
    print("\nMissing values in prepared dataset after dropping null values:")
    print(df_ml.isnull().sum())

    # Preparing features and target from the prepared data
    X = df_ml[["P", "D", "µ", "µP", "µPD", "a", "D/a"]]
    Y1 = df_ml["DE(N-mm)"]
    Y2 = df_ml["Regime"]

    # Plotting feature relationships using Pearson's correlation coefficient
    plt.figure(figsize=(16, 10))
    sns.heatmap(X.corr(), annot=True, cmap="plasma")
    plt.title("Feature Correlation", weight="bold")
    plt.show()


except FileNotFoundError:
    print(f"Dataset was not found at: {path.resolve()}")
except pd.errors.ParserError as error:
    print(f"Error while parsing the dataset: {error}")
