
import numpy as np
import networkx as nx




def codeUnrootedTree(tree):
    '''
    :param tree: is a tree encoded in NetworkX library
    :return: a binary string that identify the represent the graph
    usgae:
        G = nx.random_tree(189,2)
        co=codeUnrootedTree(G)
        print("final code")
        print(co)
    '''
    center = nx.center(tree)
    center=center[0]
    co = codeUnrootedTreeHelper(tree, center)
    return co


def codeUnrootedTreeHelper(graph,node):
    if (nx.degree(graph,node) == 0):
        return ""
    else:
        neig = np.array(list(graph.neighbors(node)))
        Gcopy = graph
        Gcopy = nx.Graph(Gcopy)
        Gcopy.remove_node(node)
        a=""
        for i in neig:
            reachNodes = nx.descendants(graph, i)
            reachNodesArray = np.array(list(reachNodes))
            reachNodesArray = np.append(i, reachNodesArray)
            H = Gcopy.subgraph(reachNodesArray)
            b = "0" + codeUnrootedTreeHelper(H, i) + "1"
            a=a+b
        return a











