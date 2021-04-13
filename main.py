import pandas as pd
import time as time
from matplotlib import pyplot as plt
from eventgraphs import EventGraph
from eventgraphs.analysis import calculate_motif_distribution
from datetime import datetime
import PySimpleGUI as sg


# TODO: Take the frequency of a sentiment motif (- or +) over a strict period of time. So it is not allowed for a
#  frequency motif to take more than x timestamps.
#  Plot motif distribution for delta:
#  - 1 hour
#  - 12 hours
#  - 1 day
#  - 2 days
#  - 7 days
#  - infinite
#  Compare the distributions and come up with a logical explanation for the differences, if any.


def temp_motifs(df):
    print("Retrieving timestamps...")
    timestamps_data = df['TIMESTAMP']
    date_times = []
    for timestamp in timestamps_data:
        datetime_object = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        date_times.append(datetime_object)
    print("There are", len(date_times), "timestamps")

    start_date = min(date_times)
    end_date = max(date_times)

    print("Transform timestamps to hour differences...")
    hour_differences = []
    for timestamp in date_times:
        time_diff = abs((timestamp - start_date).days) * 24
        hour_differences.append(time_diff)
    print("Biggest difference is", abs((end_date - start_date).days) * 24, "hours")

    print("Building new DataFrame...")
    del df['TIMESTAMP']
    df['time'] = hour_differences
    df = df.rename(columns={'SOURCE_SUBREDDIT': 'source', 'TARGET_SUBREDDIT': 'target'})
    print(df)

    print("Create event graphs...")
    # deltas = [1, 12, 24, 48, 7 * 24, abs((end_date - start_date).days) * 24]
    EG = EventGraph.from_pandas_eventlist(df, graph_rules='teg')
    EG.event_graph_rules['delta_cutoff'] = 1
    EG.build(verbose=True)

    print("Calculate edge motifs...")
    EG.calculate_edge_motifs(edge_type='type', condensed=False)
    motif_distribution = calculate_motif_distribution(EG)

    print("Motif distribution: \n", motif_distribution, '\n')
    print("Plotting results...")
    motif_distribution.plot(kind='bar', ylim=(0, 0.5))
    plt.title("Delta 1 hour")
    plt.show()


if __name__ == "__main__":
    start_time = time.time()

    print("Loading in data...")
    body_data = pd.read_csv("data/soc-redditHyperlinks-body.tsv", sep='\t')
    title_data = pd.read_csv("data/soc-redditHyperlinks-title.tsv", sep='\t')
    df = body_data.append(title_data, ignore_index=True)

    print("Removing unused columns...")
    del df['PROPERTIES']
    del df['POST_ID']
    del df['LINK_SENTIMENT']

    temp_motifs(df)

    print("Program took", str(round(time.time() - start_time, 1)), "seconds")
    sg.popup('The program is done!')

