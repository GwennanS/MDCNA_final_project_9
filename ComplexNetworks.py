#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import networkx as nx

if __name__ == "__main__":
    bodyData = pd.read_csv("soc-redditHyperlinks-body.tsv", sep='\t')
    titleData = pd.read_csv("soc-redditHyperlinks-title.tsv", sep='\t')
    df = bodyData.append(titleData, ignore_index=True)
    del df['PROPERTIES']


half_rows = int(len(df)/2)
half_rows

df = df.iloc[half_rows:,:]

g = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT')



Graph_n = nx.convert_node_labels_to_integers(g)


edges = list(Graph_n.edges())


print(len(Graph_n.nodes),len(Graph_n.edges))
for edge in edges:
    print(edge[0],edge[1])





