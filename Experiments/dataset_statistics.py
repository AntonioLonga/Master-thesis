import numpy as np
import load_data as ld
import networkx as nx
import utilities
import matplotlib.pyplot as plt
from tabulate import tabulate


def summary_load(name,pubchem=False):
    print("DATASET: \t "+str(name))
    if (pubchem == False):
        graphs,labels = ld.load_data(name)
    else:
        graphs,labels = ld.load_data_pubchem(name)
    summary_graphs_density(graphs,labels)
    summary_number_nodes_edges_in_graphs(graphs)
    summary_graphs_degrees(graphs)


def summary(graphs,labels):
    summary_graphs_density(graphs,labels)
    summary_number_nodes_edges_in_graphs(graphs)
    summary_graphs_degrees(graphs)


def summary_graphs_density(graphs,labels):
    
    densities = graphs_density(graphs)
    
    A = np.unique(labels,return_counts=True)
    A = np.transpose(A)
    t = tabulate(A, headers=['Category', 'Count'] )
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.text(0,0.9,"Tot. n. graphs: "+str(np.sum(A[:,1])),fontsize=15)
    plt.text(0,0.5,t,fontsize=15)
    plt.axis('off')


    plt.subplot(132)
    text_min = "Min density: %.5f"%np.min(densities)
    plt.text(0, 0.9, text_min,fontsize=15)
    text_max = "Max density: %.5f"%np.max(densities)
    plt.text(0, 0.7, text_max,fontsize=15)
    plt.axis('off')

    plt.subplot(133)
    plt.hist(densities,bins=30)
    plt.title("Hist. graphs density")
    plt.show()

def graphs_density(graphs):
    densities = []
    for g in graphs:
        densities.append(nx.density(g))
    return(densities)
def summary_graphs_degrees(graphs):
    g_d = graphs_degrees(graphs)

    avg_degree = []
    min_degree = []
    max_degree = []
    for i in g_d:
        avg_degree.append(np.mean(i))
        min_degree.append(np.min(i))
        max_degree.append(np.max(i))

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    text_min = "Hist. min degree \nmean: %.2f" % np.mean(min_degree) +"\n std: %.2f" % np.std(min_degree)
    plt.title(text_min)
    plt.hist(min_degree,bins=30)
    plt.subplot(132)
    text_min = "Hist. avg degree \nmean: %.2f" % np.mean(avg_degree) +"\n std: %.2f" % np.std(avg_degree)
    plt.title(text_min)
    plt.hist(avg_degree,bins=30)
    plt.subplot(133)
    text_min = "Hist. max degree \nmean: %.2f" % np.mean(max_degree) +"\n std: %.2f" % np.std(max_degree)
    plt.title(text_min)
    plt.hist(max_degree,bins=30)
    plt.show()

def graphs_degrees(graphs):
    graphs_degrees = []
    for graph in graphs:
        degrees = [val for (node, val) in graph.degree()]
        graphs_degrees.append(degrees)
    return(graphs_degrees)



def summary_number_nodes_edges_in_graphs(graphs):
    n_nodes,n_edges = number_nodes_edges_in_graphs(graphs)

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    text_min = "Min number of nodes: "+str(np.min(n_nodes))
    plt.text(0, 0.9, text_min,fontsize=15)
    text_max = "Max number of nodes: "+str(np.max(n_nodes))
    plt.text(0, 0.7, text_max,fontsize=15)
    plt.axis('off')

    
    
    plt.subplot(132)
    plt.hist(n_nodes,bins=50)
    string_node = "mean: %.2f" %(np.mean(n_nodes))+"\nstd: %.2f"%(np.std(n_nodes))
    plt.title("Hist. n. nodes per graphs \n"+string_node)
    plt.subplot(133)
    plt.hist(n_edges,bins=50)
    string_edge = "mean: %.2f" %(np.mean(n_edges))+"\nstd: %.2f"%(np.std(n_edges))
    plt.title("Hist. n. edges per graphs \n"+string_edge)
    plt.show()

    
def number_nodes_edges_in_graphs(graphs):
    n_nodes = []
    n_edges = []
    for g in graphs:
        n_nodes.append(len(g.nodes()))
        n_edges.append(len(g.edges()))

    return (n_nodes,n_edges)

