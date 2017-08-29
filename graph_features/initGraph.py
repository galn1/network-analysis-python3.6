import networkx as nx
# import utils.gnxToGgt as ut

import matplotlib.pyplot as plt
import os;

def init_graph(draw, file_name, directed,Connected = False):
    '''
    initializes the graph with using networkx packege
    :param draw: boolean parameter- True if we want to draw the graph otherwise - False
    :param file_name: the name of the file that contains the edges of the graph
    :param directed: boolean parameter- True if the graph is directed otherwise - False
    :return: nx.Graph or nx.DiGraph in accordance with the 3rd param
    '''

    if directed == True:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    with open(file_name) as f:
        for line in f:
            (v1, v2, weight) = line.split()
            G.add_edge(v1, v2,{'weight': float(weight)})
    if draw:
        draw_graph(G, directed)
    if(Connected):
        if(directed):
            G = max(nx.weakly_connected_component_subgraphs(G), key=len)
        else:
            G = max(nx.connected_component_subgraphs(G),key = len)
    return G

def draw_graph(G, directed):
    """
    This function draws the network
    :param G: nx graph or DiGraph
    :param directed: True if we want to draw the arrows of the graph for directed graph
                     otherwise - False
    :return:
    """
    pos = nx.random_layout(G)
    nx.draw(nx.Graph(G), pos)
    nx.draw_networkx_edges(G, pos, arrows= directed)
    plt.show()



    # def init_graph(draw, graph_file):
    #     file_name = os.getcwd() + r"roi-graph.txt"
    #     f = open(file_name, 'r')
    #     lines = f.read();
    #     lst = lines.split('\n')
    #     edges = []
    #     for x in [row.split(',') for row in lst]:
    #         if (len(x) == 3):  # for creation of weigthed graph
    #             temp = (float(x[0]), float(x[1]), {'weight': float(x[2])})
    #         else:
    #             temp = ((float(x[0]), float(x[1]), {'weight': 1}))
    #         edges.append(temp)
    #     gnx = nx.DiGraph()
    #     for e in edges:
    #         if (e[0] == -1 and e[1] == -1):
    #             break;
    #         gnx.add_edges_from([e])
    #     ggt = ut.nx2gt(gnx);
    #     # drawing the graph
    #     if (draw):
    #         nx.draw_networkx(gnx)
    #         plt.savefig('graph.png')
    #     return [ggt, gnx];
    #
    # def init_graph_networkx(draw=False, directed=True):
    #     file_name = os.getcwd() + r"\data\roi-graph.txt"
    #     f = open(file_name, 'r')
    #     lines = f.read();
    #     lst = lines.split('\n')
    #     edges = []
    #     for x in [row.split(',') for row in lst]:
    #         if (len(x) == 3):  # for creation of weigthed graph
    #             temp = (float(x[0]), float(x[1]), {'weight': float(x[2])})
    #         else:
    #             temp = ((float(x[0]), float(x[1]), {'weight': 1}))
    #         edges.append(temp)
    #     print(edges)
    #     if (directed):
    #         gnx = nx.DiGraph()
    #     else:
    #         gnx = nx.Graph()
    #     for e in edges:
    #         if (e[0] == -1 and e[1] == -1):
    #             break;
    #         gnx.add_edges_from([e])
    #     # drawing the graph
    #     if (draw):
    #         nx.draw_networkx(gnx)
    #         plt.savefig('graph.png')
    #     return gnx;






