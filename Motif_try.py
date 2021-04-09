import itertools
import re
import time as time
import community as community_louvain
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from eventgraphs import EventGraph
from eventgraphs.analysis import calculate_motif_distribution
from eventgraphs.plotting import plot_barcode
import numpy as np

import pandas as pd
import networkx as nx

def temp_motifs(df):
    #print("Transforming timestamps...")
    timestamps_data = df['TIMESTAMP']
    date_regex = '[0-9]+-[0-9]+-[0-9]+'
    timestamps = []
    for timestamp in timestamps_data:
        date = re.search(date_regex, timestamp)
        if date:
            timestamps.append(int(date.group(0).replace('-', '')))
    #print("There are", len(timestamps), "timestamps")

    #print("Building new DataFrame...")
    del df['TIMESTAMP']
    df['time'] = timestamps
    #print(df)
    try:
        #print("Create event graph...")
        EG = EventGraph.from_pandas_eventlist(df, graph_rules='teg')
        #print(EG)
        EG.build(verbose=True)
        #print(EG)
        #print(EG.eg_edges.head())
        EG.calculate_edge_motifs(edge_type='type', condensed=False)
        #print(EG.eg_edges)

        # print("Plotting results...")
        motif_distribution = calculate_motif_distribution(EG)
        #print(type(EG.eg_edges.motif.value_counts()))
        # motif_distribution.nlargest().plot(kind='bar', ylim=(0, 0.5))
        # plt.show()
        return EG.eg_edges.motif.value_counts()
    except:
        return {}

def motifNode(gr, Gdf, node):
    ## We define each S* motif as a directed graph in networkx
    motifs = {
        'S1': nx.DiGraph([(1, 2), (2, 3)]),
        'S2': nx.DiGraph([(1, 2), (1, 3), (2, 3)]),
        'S3': nx.DiGraph([(1, 2), (2, 3), (3, 1)]),
        'S4': nx.DiGraph([(1, 2), (3, 2)]),
        'S5': nx.DiGraph([(1, 2), (1, 3)])
    }
    mo = motifs

    """Counts motifs in a directed graph
    :param gr: A ``DiGraph`` object
    :param mo: A ``dict`` of motifs to count
    :returns: A ``dict`` with the number of each motifs, with the same keys as ``mo``
    This function is actually rather simple. It will extract all 3-grams from
    the original graph, and look for isomorphisms in the motifs contained
    in a dictionary. The returned object is a ``dict`` with the number of
    times each motif was found.::
        >>> print mcounter(gr, mo)
        {'S1': 4, 'S3': 0, 'S2': 1, 'S5': 0, 'S4': 3}
    """
    # This function will take each possible subgraphs of gr of size 3, then
    # compare them to the mo dict using .subgraph() and is_isomorphic

    # This line simply creates a dictionary with 0 for all values, and the
    # motif names as keys

    mcount = dict(zip(mo, list(map(int, np.zeros(len(mo))))))
    nodes = gr.nodes()
    print(nodes)
    print(node)
    # We use iterools.product to have all combinations of three nodes in the
    # original graph. Then we filter combinations with non-unique nodes, because
    # the motifs do not account for self-consumption.

    triplets = list(itertools.product(*[nodes, nodes, nodes]))
    triplets = [trip for trip in triplets if (len(list(set(trip))) == 3 and node in list(set(trip)))]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]
    # The for each each of the triplets, we (i) take its subgraph, and compare
    # it to all fo the possible motifs

    temp_motif = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for trip in u_triplets:
        Tripdf = Gdf[Gdf.source.isin(trip)]
        Tripdf = Tripdf[Tripdf.target.isin(trip)]
        if not Tripdf.empty:
            new = temp_motifs(Tripdf)
        #print(new.items())
            for key, value in new.items():
            #print(key == {'●|○': 45, '○|●': 7, '○|+': 4, '+|●': 4, '●|+': 3, '+|○': 2})
                if key == '●|○':
                    temp_motif[0] = temp_motif[0] + value
                if key == '○|●':
                    temp_motif[1] = temp_motif[1] + value
                if key == '○|+':
                    temp_motif[2] = temp_motif[2] + value
                if key == '+|●':
                    temp_motif[3] = temp_motif[3] + value
                if key == '●|+':
                    temp_motif[4] = temp_motif[4] + value
                if key == '+|○':
                    temp_motif[5] = temp_motif[5] + value
    print(temp_motif)
    factor = 1.0 / sum(temp_motif.values())
    for k in temp_motif:
        temp_motif[k] = temp_motif[k] * factor
    print(temp_motif)

    for trip in u_triplets:
        sub_gr = gr.subgraph(trip)
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), motifs.keys()))
        match_keys = [list(mo.keys())[i] for i in range(len(mo)) if mot_match[i]]
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1
    print(mcount)
    return mcount


def louvain(df):
    print("Transforming labels...")
    g = nx.from_pandas_edgelist(df, 'source', 'target')
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

if __name__ == '__main__':
    bodyData = pd.read_csv("soc-redditHyperlinks-body.tsv", sep='\t')
    titleData = pd.read_csv("soc-redditHyperlinks-title.tsv", sep='\t')
    df = bodyData.append(titleData, ignore_index=True)

    G = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', ['POST_ID', 'TIMESTAMP', 'LINK_SENTIMENT'], create_using=nx.DiGraph)
    #print(nx.info(G))


    mapping = dict(zip(G.nodes(), range(0, G.number_of_nodes())))


    # intresting nodes talked about
    ranking_indegree = [None] * G.number_of_nodes()
    rank = 0
    degree_sort = sorted(G.in_degree(), key=lambda x: -x[1])
    for tuple_degree in degree_sort:
        ranking_indegree[rank] = tuple_degree[0]
        rank = rank + 1
    # print(degree_sort[:20])
    # print("ranking_indegree: ", ranking_indegree[:20])
    # print(mapping.get(ranking_indegree[0]))
    # print(list(list(G.adjacency())[mapping.get(ranking_indegree[0])][1].keys()))
    #G_in_degree = G.subgraph(list(list(G.adjacency())[mapping.get(ranking_indegree[0])][1].keys()))
    #G = G.subgraph(ranking_indegree[:820])
    #print(nx.info(G_in_degree))
    #G.name = "G_in_degree"
    # interesting nodes that do the talking
    ranking_outdegree = [None] * G.number_of_nodes()
    rank = 0
    degree_sort = sorted(G.out_degree(), key=lambda x: -x[1])
    for tuple_degree in degree_sort:
        ranking_outdegree[rank] = tuple_degree[0]
        rank = rank + 1
    #print(degree_sort[:20])
    #print("ranking_outdegree: ", ranking_outdegree[:20])
    #G_out_degree = G.subgraph(list(list(G.adjacency())[mapping.get(ranking_outdegree[0])][1].keys()))
    #G = G.subgraph(ranking_outdegree[:708])
    G = G.subgraph(ranking_outdegree[:100])
    #print(nx.info(G_out_degree))
    G.name = "G_out_degree"

    Gdf = df.rename(columns={'SOURCE_SUBREDDIT': 'source', 'TARGET_SUBREDDIT': 'target'})
   # print(Gdf.shape)
    Gdf = Gdf[Gdf.source.isin(ranking_outdegree[:100])]
    Gdf = Gdf[Gdf.target.isin(ranking_outdegree[:100])]
    #print(Gdf.shape)
    print(G)
    motifNode(G, Gdf, ranking_outdegree[0])

    # Graph_n = nx.convert_node_labels_to_integers(G)
    #
    # edges = list(Graph_n.edges())
    #
    # print(len(Graph_n.nodes), len(Graph_n.edges))
    # for edge in edges:
    #     print(edge[0], edge[1])

    # %%

    # G2 = nx.from_pandas_edgelist(df, source='SOURCE_SUBREDDIT', target='TARGET_SUBREDDIT', edge_attr=['LINK_SENTIMENT'])
    #
    # # %%
    #
    # print(nx.info(G2))
    # print("Average Degree ", np.mean(np.asarray(list(dict(G2.degree()).values()))))
    # print("Standard Deviation of Degree", np.std(np.asarray(list(dict(G2.degree()).values()))))
    #



    #print(G.edges(ranking_degree[:2]))
    # print(G.edges().data(keys="LINK_SENTIMENT"))
    # Gsmaller1 = nx.from_edgelist(G.edges().data("LINK_SENTIMENT"))
    # print(nx.info(Gsmaller1))
    # Gsmaller2 = nx.from_edgelist(G.edges().data("LINK_SENTIMENT", -1))
    # # print(nx.info(Gsmaller2))
    # print((Gsmaller1.edges().data()))

    #
    # mapping = dict(zip(G.nodes(), range(0, G.number_of_nodes())))
    #
    # mapping.get(ranking_degree[0])
    # print(mapping.get(ranking_degree[0]))
    #
    # print(list(list(G.adjacency())[mapping.get(ranking_degree[0])][1].keys()))

    # half_rows = int(len(df) / 2)
    # half_rows
    #
    # df = df.iloc[half_rows:, :]
    #
    # g = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT')
    #
    # Graph_n = nx.convert_node_labels_to_integers(g)
    #
    # edges = list(Graph_n.edges())
    #

    # av = (float(sum(dict(G.degree()).values())) / float(G.number_of_nodes()))
    # to_remove = [n for n in G.nodes() if G.degree(n) <=av]
    #
    # G.remove_nodes_from(to_remove)
    #
    # print(nx.info(G))

    #
    # to_keep = [n for n in outdeg if outdeg[n] != 1]
    # G.subgraph(to_keep)







    #print(xx.edges().data("LINK_SENTIMENT", 1))
    #print(nx.adjacency_matrix(xx))
    #del df['PROPERTIES']

    # pp = pathpy.TemporalNetwork()
    # ff = df.values.tolist()
    # for item in ff:
    #     pp.add_edge(item[0], item[1], item[3])
    # nn = pathpy.Network.from_temporal_network(pp)
    # print(pp)

    # G = xx
    #
    # degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    # dmax = max(degree_sequence)
    #
    # plt.loglog(degree_sequence, "b-", marker="o")
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    #
    # # draw graph in inset
    # plt.axes([0.45, 0.45, 0.45, 0.45])
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # pos = nx.spring_layout(Gcc)
    # plt.axis("off")
    # nx.draw_networkx_nodes(Gcc, pos, node_size=20)
    # nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
    # plt.show()


