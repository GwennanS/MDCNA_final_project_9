import pandas as pd
import time as time
from matplotlib import pyplot as plt
from eventgraphs import EventGraph
from eventgraphs.analysis import calculate_motif_distribution
from datetime import datetime
import PySimpleGUI as sg


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
    print("Start date:", start_date)
    print("End date:", end_date)

    print("Transform timestamps to hour differences...")
    hour_differences = []
    for timestamp in date_times:
        time_diff = abs((timestamp - start_date).seconds) // 3600 + abs((timestamp - start_date).days) * 24
        hour_differences.append(time_diff)
    diff = abs((end_date - start_date).seconds) // 3600 + abs((end_date - start_date).days) * 24
    print("Biggest difference is", diff, "hours")

    print("Building new DataFrame...")
    del df['TIMESTAMP']
    df['time'] = hour_differences
    df = df.rename(columns={'SOURCE_SUBREDDIT': 'source', 'TARGET_SUBREDDIT': 'target'})
    print(df)

    print("Create event graphs...")
    deltas = [1, 48, abs((end_date - start_date).days) * 24]
    for i in deltas:
        print("Delta =", i)
        EG = EventGraph.from_pandas_eventlist(df, graph_rules='teg')
        EG.event_graph_rules['delta_cutoff'] = i
        EG.build(verbose=True)

        print("Calculate edge motifs...")
        EG.calculate_edge_motifs(edge_type='type', condensed=False)
        motif_distribution = calculate_motif_distribution(EG)

        print("Motif distribution: \n", motif_distribution, '\n')
        print("Plotting results...")
        motif_distribution.plot(kind='bar', ylim=(0, 0.5))
        delta = str(i)
        plt.title("Delta " + delta + " hours (-)")
        plt.show()


if __name__ == "__main__":
    start_time = time.time()

    print("Loading in data...")
    body_data = pd.read_csv("data/soc-redditHyperlinks-body.tsv", sep='\t')
    title_data = pd.read_csv("data/soc-redditHyperlinks-title.tsv", sep='\t')
    df = body_data.append(title_data, ignore_index=True)

    # df = df[df['LINK_SENTIMENT'] == -1]

    print("Removing unused columns...")
    del df['PROPERTIES']
    del df['POST_ID']
    del df['LINK_SENTIMENT']

    temp_motifs(df)

    print("Program took", str(round(time.time() - start_time, 1)), "seconds")
    sg.popup('The program is done!')

