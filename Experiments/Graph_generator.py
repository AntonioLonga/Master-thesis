import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
class Graph_generator:

    
    '''
    Generate similar graphs, take in input a graph, and produce a list of graphs
    '''
    def gen_similar_graphs(g,depth=1):
        #delete one edge from the original grpah
        graphs = Graph_generator.gen_delete_edge([g])
        graphs_change_node_att = Graph_generator.gen_change_node_label(graphs)
        
        for g in graphs_change_node_att:
            graphs.append(g)
            
        graphs_change_edge_att = Graph_generator.gen_change_edge_label(graphs)
        for g in graphs_change_edge_att:
            graphs.append(g)
    
        depth = depth - 1 
        
        # for each graph in graphs_change_node_att, repeat the procedure:
        # 1) remove one arch, chagne all the labels one at each time

        for i in range (0,depth):
            graph2 = Graph_generator.gen_delete_edge(graphs_change_node_att)
            graph3 = Graph_generator.gen_change_node_label(graph2)

            for i in graph2:
                graphs.append(i)
            for i in graph3:
                graphs.append(i)
            
            graph4 = Graph_generator.gen_change_edge_label(graphs)
            for i in graph4:
                graphs.append(i)


        return(graphs)
    def gen_change_edge_label(graphs):
        graph_chagne_edge = []
        n_nodes = len(graphs[0].nodes())
        for t in graphs:
            for n in t.edges(data=True):
                t_tmp = t.copy()
                lab = np.random.randint(n_nodes)
                t_tmp.add_edge(n[0],n[1],label=lab)
                graph_chagne_edge.append(t_tmp)
        return (graph_chagne_edge)

    '''
    From a given list of grphs, change randomly the label of the node.
    at each etaration only ONLY one label is changed
    '''
    def gen_change_node_label(graphs):

        graphs_change_node_att=[]
        n_nodes = len(graphs[0].nodes())
        for t in graphs:
            for n in t.nodes():
                t_tmp = t.copy()
                lab = np.random.randint(n_nodes)
                t_tmp.add_node(n,label=lab)
                graphs_change_node_att.append(t_tmp)

        return (graphs_change_node_att)

    '''
    From a given grpah delete one edge, at each time verify that the graph is sitll connected
    return an array of graphs
    '''
    def gen_delete_edge(graphs):
        
        graphs_deleted_edge = []
        for g in graphs:
            for e in g.edges():
                g_tmp = g.copy()
                g_tmp.remove_edge(e[0],e[1])
                if (nx.is_connected(g_tmp)):
                    graphs_deleted_edge.append(g_tmp)

        return(graphs_deleted_edge)


    '''
    Generate a CONNECTED graph with random node and edge labels
    return a nx graph
    '''
    def gen_graph(n_nodes,prob_edge, seed=None):
        g = nx.random_geometric_graph(n_nodes,prob_edge,seed=seed)
        for n in g.nodes():
            lab = np.random.randint(n_nodes)
            g.add_node(n,label=lab)
            del g.node[n]['pos']
        for e in g.edges():
            att1 = np.random.randint(n_nodes)
            g.add_edge(e[0],e[1],label=att1)
                      
        if (nx.is_connected(g)):
            return(g)
        else:
            if (seed == None):
                seed=np.random.randint(100)
            else:
                seed = seed + 1
            return(Graph_generator.gen_graph(n_nodes,prob_edge,seed))


    def myplot(G,n_label='label'):
        pos = nx.spring_layout(G,seed =4)

        nx.draw(G, pos)
        node_labels = nx.get_node_attributes(G,n_label)
        nx.draw_networkx_labels(G, pos, labels = node_labels)
        edge_labels = nx.get_edge_attributes(G,'')
        nx.draw_networkx_edge_labels(G, pos, labels = edge_labels)

        plt.show()