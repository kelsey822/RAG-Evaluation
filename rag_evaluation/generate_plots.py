"""Script to generate relevant plots for the retrieval data.
"""

import pandas as pd
import plotly.express as px


def create_violin(df, x: str, y: str, title: str):
    fig = px.violin(df, y="value", color=x, box=True, points="all", title=title)
    fig.update_layout(yaxis_title=y, xaxis_title=x, boxmode="overlay")
    fig.show()


def create_bar(df, x: str, y: str, color: str, title: str):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
    fig.show()


def create_scatter(df, x: str, y: str, color=None, symbol=None):
    fig = px.scatter(df, x=x, y=y, color=color, symbol=symbol)
    fig.show()


def create_confusion_matrix(df, group_1: str, group_2: str):
    cm = ((df.groupby([group_2, group_1])).size()).unstack(fill_value=0)
    cm = cm.reindex([1, 0], axis=0).reindex([1, 0], axis=1)
    print(cm)


def filter_data_by_source(df, columns: list):
    """Creates source dataframe from the main dataframe given a list of columns to filter."""
    df = df.iloc[:, columns]
    return df


def transform_source_df(df, var_name: str, value_name: str):
    """Group long data together for ease of plotting."""
    # transform wide format to long
    # df = df.melt(var_name, value_name)

    # convert value to string to use discrete coloring
    df[value_name] = df[value_name].astype(str)

    # stack the data by grouping by relevance and value and count them
    df = df.groupby([var_name, value_name]).size().reset_index(name="count")
    return df


def generate_source_plots(
    df, source_df: str, columns: list, group_1: str, group_2: str, plot_name: str
):
    """Generates the bar graph and confusion matrix for a given source after manipulating the data."""
    # transform the data frame
    source_df = filter_data_by_source(df, columns)  # wide data
    long_source_df = transform_source_df(
        source_df.melt(var_name="relevance", value_name="value"), "relevance", "value"
    )  # long data

    create_confusion_matrix(source_df, group_1, group_2)
    create_bar(long_source_df, "relevance", "count", "value", plot_name)


if __name__ == "__main__":
    # read and clean the raw retrieval data
    retrieved_df = pd.read_csv("./data/data.csv")  # convert the csv to a df
    retrieved_df = retrieved_df.loc[
        retrieved_df["source 1 human relevance"] != -1
    ]  # get rid of the -1 rows
    retrieved_df = retrieved_df.iloc[
        :, 7:
    ]  # get rid of the queries and sources (wide data)

    # generate human retrieval metric plots (i.e on a scale of 0-4)
    human_relevance_df = retrieved_df.iloc[:, [0, 1, 2, 3, 4]]
    long_human_relevance_df = transform_source_df(
        human_relevance_df.melt(var_name="relevance", value_name="value"),
        "relevance",
        "value",
    )  # long data
    create_bar(
        long_human_relevance_df, "relevance", "count", "value", "human_relevance"
    )

    # generate plots for source 1
    generate_source_plots(
        retrieved_df,
        "source1_df",
        [5, 10],
        "source 1 human binary relevance",
        "source 1 generated binary relevance",
        "source 1",
    )

    # generate plots for source 2
    generate_source_plots(
        retrieved_df,
        "source2_df",
        [6, 11],
        "source 2 human binary relevance",
        "source 2 generated binary relevance",
        "source 2",
    )

    # generate plots for source 3
    generate_source_plots(
        retrieved_df,
        "source3_df",
        [7, 12],
        "source 3 human binary relevance",
        "source 3 generated binary relevance",
        "source 3",
    )

    # generate plots for source 4
    generate_source_plots(
        retrieved_df,
        "source4_df",
        [8, 13],
        "source 4 human binary relevance",
        "source 4 generated binary relevance",
        "source 4",
    )

    # generate plots for source 5
    generate_source_plots(
        retrieved_df,
        "source5_df",
        [9, 14],
        "source 5 human binary relevance",
        "source 5 generated binary relevance",
        "source 5",
    )

    # read and clean the generated retrieval metrics (e.g. precision @k)
    normal_retrieved_df = pd.read_csv("./data/normalized_metrics.csv")
    normal_retrieved_unswapped = normal_retrieved_df.iloc[:, 1:]
    normal_retrieved = normal_retrieved_unswapped.melt(
        var_name="metric", value_name="value"
    )

    # plot the normalized retrieval metrics
    create_violin(normal_retrieved, "metric", "value", "retrieval metrics")

    # read, clean, and transform the llm data
    llm_df = (pd.read_csv("./data/llm_responses_data.csv")).iloc[
        :, 2:-1
    ]  # get just the metrics from the csv file

    # convert llm data to long data
    long_llm_df = llm_df.melt(var_name="statistic", value_name="value")
    long_llm_df["value"] = long_llm_df["value"].astype(str)
    long_llm_df = (
        long_llm_df.groupby(["statistic", "value"]).size().reset_index(name="count")
    )

    # plot the llm repsonse data
    create_bar(long_llm_df, "statistic", "count", "value", "llm response")
