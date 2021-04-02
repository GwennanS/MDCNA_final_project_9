import re
import pandas as pd
import time as time
import networkx as nx
import community as community_louvain
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from eventgraphs import EventGraph
from eventgraphs.analysis import calculate_motif_distribution
from eventgraphs.plotting import plot_barcode

def temp_motifs(df):
    print("Transforming timestamps...")
    timestamps_data = df['TIMESTAMP']
    date_regex = '[0-9]+-[0-9]+-[0-9]+'
    timestamps = []
    for timestamp in timestamps_data:
        date = re.search(date_regex, timestamp)
        if date:
            timestamps.append(int(date.group(0).replace('-', '')))
    print("There are", len(timestamps), "timestamps")

    print("Building new DataFrame...")
    del df['TIMESTAMP']
    df['time'] = timestamps
    df = df.rename(columns={'SOURCE_SUBREDDIT': 'source', 'TARGET_SUBREDDIT': 'target'})
    print(df)

    print("Create event graph...")
    EG = EventGraph.from_pandas_eventlist(df, graph_rules='teg')
    print(EG)
    EG.build(verbose=True)
    print(EG)
    print(EG.eg_edges.head())
    EG.calculate_edge_motifs(edge_type='type', condensed=False)
    print(EG.eg_edges)

    print("Plotting results...")
    motif_distribution = calculate_motif_distribution(EG)
    print(motif_distribution)
    motif_distribution.nlargest().plot(kind='bar', ylim=(0, 0.5))
    plt.show()


def louvain(df):
    print("Transforming labels...")
    g = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT')
    G = nx.convert_node_labels_to_integers(g)
    # compute the best partition
    print("Finding best partitions...")
    partition = community_louvain.best_partition(G)

    print("Plotting...")
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    # df = nx.to_pandas_edgelist(g)
    # print("There are", len(df), "edges")


if __name__ == "__main__":
    start_time = time.time()

    print("Loading in data...")
    body_data = pd.read_csv("data/soc-redditHyperlinks-body.tsv", sep='\t')
    # title_data = pd.read_csv("data/soc-redditHyperlinks-title.tsv", sep='\t')
    df = body_data.head(50000)  # .append(title_data, ignore_index=True)
    del df['PROPERTIES']
    del df['POST_ID']
    del df['LINK_SENTIMENT']

    louvain(df)
    # temp_motifs(df)

    print("Program took", str(round(time.time() - start_time, 1)), "seconds")
