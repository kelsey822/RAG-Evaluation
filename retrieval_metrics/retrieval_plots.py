"""Script to generate relevant plots for the retrieval data.
"""

import csv

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_violin(df, x, y):
    fig = px.violin(
        df, y="value", color=x, box=True, points="all", title="Violin Plot of Metrics"
    )
    fig.update_layout(yaxis_title=y, xaxis_title=x, boxmode="overlay")
    fig.show()


def create_bar(df, x, y, color, title):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
    fig.show()


if __name__ == "__main__":
    # read the data
    retrieved_df = pd.read_csv("data.csv")
    normal_retrieved_df = pd.read_csv("normalized_metrics.csv")

    # clean the data
    filtered_data = retrieved_df.loc[retrieved_df["human_relevance1"] != -1]
    normal_retrieved = normal_retrieved_df.iloc[:, 1:]
    normal_retrieved = normal_retrieved.melt(var_name="metric", value_name="value")

    # get the medical query data
    medical_data = filtered_data.loc[filtered_data["medical"] == "1"]

    # get the main retrieval data
    swapped_data = filtered_data.iloc[:, 7:]
    main_data = filtered_data.melt(var_name="relevance", value_name="value")

    # create dfs for each retrieved source
    human_relevance_df = filtered_data.iloc[:, [7, 8, 9, 10, 11]]
    hr_df = human_relevance_df.melt(var_name="relevance", value_name="value")
    hr_df = hr_df.groupby(["relevance", "value"]).size().reset_index(name="count")

    source1_df = filtered_data.iloc[:, [12, 17]]
    source1_df = source1_df.melt(var_name="relevance", value_name="value")
    source1_df = (
        source1_df.groupby(["relevance", "value"]).size().reset_index(name="count")
    )

    source2_df = filtered_data.iloc[:, [13, 18]]
    source2_df = source2_df.melt(var_name="relevance", value_name="value")
    source2_df = (
        source2_df.groupby(["relevance", "value"]).size().reset_index(name="count")
    )

    source3_df = filtered_data.iloc[:, [14, 19]]
    source3_df = source3_df.melt(var_name="relevance", value_name="value")
    source3_df = (
        source3_df.groupby(["relevance", "value"]).size().reset_index(name="count")
    )

    source4_df = filtered_data.iloc[:, [15, 20]]
    source4_df = source4_df.melt(var_name="relevance", value_name="value")
    source4_df = (
        source4_df.groupby(["relevance", "value"]).size().reset_index(name="count")
    )

    source5_df = filtered_data.iloc[:, [16, 21]]
    source5_df = source5_df.melt(var_name="relevance", value_name="value")
    source5_df = (
        source5_df.groupby(["relevance", "value"]).size().reset_index(name="count")
    )

    # plot the normalized retrieval metrics
    create_violin(normal_retrieved, "metric", "value")

    # plot the binary relevance metrics
    create_bar(source1_df, "relevance", "count", "value", "Source 1")
    create_bar(source2_df, "relevance", "count", "value", "Source 2")
    create_bar(source3_df, "relevance", "count", "value", "Source 3")
    create_bar(source4_df, "relevance", "count", "value", "Source 4")
    create_bar(source5_df, "relevance", "count", "value", "Source 5")

    # read the data for llm responses
    llm_df = (pd.read_csv("llm_responses_data.csv")).iloc[:, 2:-1]
    llm_df = llm_df.melt(var_name="statistic", value_name="value")
    llm_df = llm_df.groupby(["statistic", "value"]).size().reset_index(name="count")
    print(llm_df)

    # plot the llm repsonse data
    create_bar(llm_df, "statistic", "count", "value", "llm response")
