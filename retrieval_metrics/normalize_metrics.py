"""Normalizes data using max min scaling so all the metrics can be easily compared.
"""

import csv

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == "__main__":
    # open the file with the metrics
    df = pd.read_csv("metrics.csv")

    # get the variables
    queries = df[["prompt"]]
    metrics = df.drop(columns=["prompt"])

    # normalize using min max scaling
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(metrics)

    # add normalized data to a data frame
    min_max = pd.concat(
        [queries, pd.DataFrame(normalized_metrics, columns=metrics.columns)], axis=1
    )

    # write the df to a new csv file
    min_max.to_csv("normalized_metrics.csv", index=False)
