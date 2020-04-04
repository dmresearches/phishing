

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community
import collections

#import data 
data = pd.read_csv("F:\\D G\\py\\phishing.csv")

#cut data to tree part(all, legitimate, phishing) 
all_of_data = data.copy()
del all_of_data['Result']

legitimate_of_data = data[data.Result == 1]
del legitimate_of_data['Result']

phishing_of_data = data[data.Result == -1]
del phishing_of_data['Result']
del phishing_of_data['Prefix_Suffix'] #prefix_suffix delete because prefix_suffix dose not change 

# compute correlation 
all_corr = all_of_data.corr().values
legitimat_corr = legitimate_of_data.corr().values
phishing_corr = phishing_of_data.corr().values

# method for compute simmilarity input=correlation matrix, output=simmilarity matrix without diagonal
def similar(data):
    c = np.zeros((data.shape[0],data.shape[1]))
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            item = data[i,j]
            sigma = 1
            distance = np.sqrt(2*(1-item))
            similarity = np.exp(-distance/(sigma**2))
            c[i,j] = similarity
            np.fill_diagonal(c,0)
    return(c)

sim_one = similar(all_corr)
sim_two = similar(legitimat_corr)
sim_tree = similar(phishing_corr)

# similarity matrix change to networks
A = nx.from_numpy_matrix(sim_one)
B = nx.from_numpy_matrix(sim_two)
C = nx.from_numpy_matrix(sim_tree)

# best partition for networks
partition_one = community.best_partition(A)
partition_two = community.best_partition(B)
partition_tree = community.best_partition(C)

# take label from data to show on plots
label_one = dict(zip((list(range(0,data.shape[0]))),(all_of_data.columns.values)))
label_two = dict(zip((list(range(0,data.shape[0]))),(legitimate_of_data.columns.values)))
label_tree = dict(zip((list(range(0,data.shape[0]))),(phishing_of_data.columns.values)))

# drawing graph A
size_one = float(len(set(partition_one.values())))
pos_one = nx.spring_layout(A)
count_one = 0.

for com_one in set(partition_one.values()) :
    count_one = count_one + 1.
    list_nodes_one = [nodes for nodes in partition_one.keys()
                                if partition_one[nodes] == com_one]
    nx.draw_networkx_nodes(A, pos_one, list_nodes_one, node_size = 1000,
                               node_color = str(count_one / size_one))

nx.draw_networkx_edges(A,pos_one, alpha=0.5, edge_color='c')
nx.draw_networkx_labels(A,pos_one, labels=label_one, font_size=30, font_family='sans-serif')

plt.figure()

# drawing graph B
size_two = float(len(set(partition_two.values())))
pos_two = nx.spring_layout(B)
count_two = 0.

for com_two in set(partition_two.values()) :
    count_two = count_two + 1.
    list_nodes_two = [nodes for nodes in partition_two.keys()
                                if partition_two[nodes] == com_two]
    nx.draw_networkx_nodes(B, pos_two, list_nodes_two, node_size = 1000,
                               node_color = str(count_two / size_two))
    
nx.draw_networkx_edges(B,pos_two, alpha=0.5, edge_color='c')
nx.draw_networkx_labels(B,pos_two, labels=label_two, font_size=30, font_family='sans-serif')

plt.figure()

# drawing graph C
size_tree = float(len(set(partition_tree.values())))
pos_tree = nx.spring_layout(C)
count_tree = 0.

for com_tree in set(partition_tree.values()) :
    count_tree = count_tree + 1.
    list_nodes_tree = [nodes for nodes in partition_tree.keys()
                                if partition_tree[nodes] == com_tree]
    nx.draw_networkx_nodes(C, pos_tree, list_nodes_tree, node_size = 1000,
                               node_color = str(count_tree / size_tree))

nx.draw_networkx_edges(C,pos_tree, alpha=0.5, edge_color='c')
nx.draw_networkx_labels(C,pos_tree, labels=label_tree, font_size=30, font_family='sans-serif')

plt.figure()

# bild dictionary for show which feature in community
one = dict(zip(list(label_one.values()),list(partition_one.values())))
two = dict(zip(list(label_two.values()),list(partition_two.values())))
tree = dict(zip(list(label_tree.values()),list(partition_tree.values())))

# method for finding commiunity 
# input = dictionary with features(keys) and partition(values) , integer (community which like to find)
# output = features in community
def find_comm(input_dict, value):
    return {k for k, v in input_dict.items() if v == value}

first_comm_one = list(find_comm(one,0))
second_comm_one = list(find_comm(one,1))
third_comm_one = list(find_comm(one,2))

first_comm_two = list(find_comm(two,0))
second_comm_two = list(find_comm(two,1))
third_comm_two = list(find_comm(two,2))

first_comm_tree = list(find_comm(tree,0))
second_comm_tree = list(find_comm(tree,1))
third_comm_tree = list(find_comm(tree,2))

# maximum spanning tree for (A,B,C)
D = nx.maximum_spanning_tree(A)
E = nx.maximum_spanning_tree(B)
F = nx.maximum_spanning_tree(C)

# method for drawing maximum spanning tree
# input = graph G , dictionary of label
# output = plot with label
def draw(G,lab):
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos, labels=lab, font_size=30)

# drawing maximum spanning trees  
draw(D,label_one)
plt.figure()    
draw(E,label_two)
plt.figure()
draw(F,label_tree)
plt.figure()

# define lists of degrees for each of maximum spanning trees
degree_sequence_one = sorted([d for n, d in D.degree()], reverse=True) 
degree_sequence_two = sorted([d for n, d in E.degree()], reverse=True) 
degree_sequence_tree = sorted([d for n, d in F.degree()], reverse=True) 

# drawing probability plot for degrees
degreeCount_one = collections.Counter(degree_sequence_one)
deg_one, cnt_one = zip(*degreeCount_one.items())
cnt_one = np.array(cnt_one)
cntt_one = cnt_one/sum(cnt_one)
plt.scatter(deg_one,cntt_one)
plt.plot(deg_one , cntt_one ,label='all of data')
plt.legend()

degreeCount_two = collections.Counter(degree_sequence_two)
deg_two, cnt_two = zip(*degreeCount_two.items())
cnt_two =np.array(cnt_two)
cntt_two = cnt_two/sum(cnt_two)

plt.scatter(deg_two,cntt_two)
plt.plot(deg_two , cntt_two , label='legitimate data' )

degreeCount_tree = collections.Counter(degree_sequence_tree)
deg_tree, cnt_tree = zip(*degreeCount_tree.items())
cnt_tree = np.array(cnt_tree)
cntt_tree = cnt_tree/sum(cnt_tree)
plt.scatter(deg_tree,cntt_tree)
plt.plot(deg_tree , cntt_tree , label='phishing data' )
plt.legend()

# difine dictionary with labels and degrees of each nodes
dat = dict(zip(list(label_one.values()),list(dict(D.degree()).values())))
leg = dict(zip(list(label_two.values()),list(dict(E.degree()).values())))
phish = dict(zip(list(label_tree.values()),list(dict(F.degree()).values())))

# method for finding hobs
# input = dictionary with labels and degree of eah nodes
# output = set of nodes that is hob
def find_hob(input_dict):
    return {k for k, v in input_dict.items() if v > 2}


w=list(find_hob(dat))
x=list(find_hob(leg))
z=list(find_hob(phish))

result_one = []
error_one = []
for u in range(1,100):
    gamma_one = u / 10
    for b in range(1,5):
        p_one = b ** (-gamma_one)
        result_one.append(p_one)
    error_one.append(np.abs(np.array(p_one) - cntt_one).sum())
res_one = np.argmin(error_one)  
res_one = (res_one +1)/100
print(res_one)

result_two = []
error_two = []
for q in range(1,100):
    gamma_two = q / 10
    for a in range(1,6):
        p_two = a ** (-gamma_two)
        result_two.append(p_two)
    error_two.append(np.abs(np.array(p_two) - cntt_two).sum())
res_two = np.argmin(error_two)  
res_two = (res_two +1)/100
print(res_two)        

result_tree = []
error_tree = []
for e in range(1,100):
    gamma_tree = e / 10
    for p in range(1,5):
        p_tree = p ** (-gamma_tree)
        result_tree.append(p_tree)
    error_tree.append(np.abs(np.array(p_tree) - cntt_tree).sum())
res_tree = np.argmin(error_tree)  
res_tree = (res_tree +1)/100
print(res_tree)
