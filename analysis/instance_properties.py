#!/usr/bin/env python3
"""
Compute SK problem instance properties as reported in the manuscript. Per instance, compute:

- Graph class
- Graph class cardinality
- Ground space
- Ground space energy
- Cost function landcape class
"""
import pickle
import os
import sys
import numpy as np
from time import time
from tabulate import tabulate
import networkx as nx
import chainer as ch

sys.path.insert(1, "../")
import _HQAOA
import qem


class GraphIt:  # Create init graph. The __next__ method returns the next SK graph in a certain sequence.
    def __init__(self, n):
        self.cgi = -1  # Init graph integer
        self.n = n
        self.ned = int((n - 1) ** 2 / 2 + (n - 1) / 2)  # Number of edges

    def i_to_bstr(
        self, i
    ):  # Turn the integer i to a bitstring in the form of an iterator. Also output its hamming weight. The bitstring is of the form e.g. [-1,1,1,...] and represents a SK graph by only specifying weights.
        get_bin = lambda x: format(x, "b").zfill(
            self.ned
        )  # Function that outputs integers in the binary representation as a string.
        st = get_bin(i)
        bstr = [int(st[j]) for j in range(len(st))]
        ham_weight = sum(bstr)
        bstr = [-(bit * 2 - 1) for bit in bstr]
        bstr = iter(bstr)
        return bstr, ham_weight

    def bstr_to_graph(self, bstr, ham_weight):
        edges = [
            (i, j, next(bstr)) for i in range(self.ned) for j in range(i + 1, self.n)
        ]
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        return g

    def __iter__(self):
        return self

    def __next__(self):
        self.cgi += 1
        bstr, ham_weight = self.i_to_bstr(self.cgi)
        bstr, ham_weight = self.i_to_bstr(self.cgi)
        g = self.bstr_to_graph(bstr, ham_weight)
        if self.cgi >= 2 ** (self.ned):
            raise StopIteration

        return g, ham_weight


def plot_graph(tup, identifier):
    g = tup[0]
    mult = tup[1]
    plt.clf()
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos)
    labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    plt.title(str(mult))
    plt.savefig("graph_plot_{}.pdf".format(identifier))


def edge_match(e1, e2):
    return e1["weight"] == e2["weight"]


def compute_classes(
    n,
):  # Compute the classes, send Hamming weights and multiplicities to st out, save graphs, Hamming weights and multiplicities to disk.
    ned = int((n - 1) ** 2 / 2 + n / 2)  # Number of edges.
    gi = GraphIt(n)
    classes = [[*next(gi), 1]]
    n_bstr = 2**ned
    i = 0
    print("Progress:")
    for g, ham_weight in gi:
        i += 1
        print(i / (n_bstr - 1) * 100, "\r", end="")  # Progress indicator.
        iso = False
        for j, tup in enumerate(
            classes
        ):  # Check if the current graph is isomorphic to any of the previously encountered class representatives.
            if ham_weight == tup[1]:
                if nx.is_isomorphic(tup[0], g, edge_match=edge_match):
                    classes[j][2] += 1
                    iso = True
                    break
        if (
            iso == False
        ):  # If it was not isomorphic to any of the class representatives, add the graph to the list of representatives.
            classes.insert(0, [g, ham_weight, 1])

    classes.sort(key=lambda x: x[2])
    # Print to st out.
    meta = [tup[1:] for tup in classes]
    all_mult = [p[1] for p in meta]
    print("[[ham_weight,mult],...]:")
    print(meta)
    assert (
        sum(all_mult) == 2**ned
    ), "Graph multiplicities do not add up to total numer of SK graphs."

    # Save to disk.
    name = "SK_graph_classes_n={}.dat".format(n)
    with open(name, "wb") as file:
        pickle.dump(classes, file, protocol=4)
    print(
        "Wrote graph classes to", name, "in the format [[nx.Graph,ham_weight,mult],...]"
    )

    ## To load back the graphs, use
    # with open('<name>.dat','rb') as file:
    #    st=pickle.load(file)

    return classes


# compute_classes(6) # gives SK_graph_classes_n=6.dat and stdout, in syntax [[ham_weight,mult],...],
# [[15, 1], [0, 1], [5, 6], [10, 6], [9, 10], [6, 10], [14, 15], [12, 15], [9, 15], [8, 15], [7, 15], [3, 15], [6, 15], [1, 15], [12, 20], [3, 20], [11, 30], [4, 30], [13, 45], [11, 45], [10, 45], [5, 45], [4, 45], [2, 45], [13, 60], [9, 60], [11, 60], [10, 60], [12, 60], [11, 60], [9, 60], [6, 60], [8, 60], [5, 60], [6, 60], [4, 60], [9, 60], [7, 60], [6, 60], [4, 60], [3, 60], [2, 60], [10, 72], [5, 72], [11, 90], [10, 90], [7, 90], [8, 90], [10, 90], [9, 90], [9, 90], [6, 90], [7, 90], [5, 90], [4, 90], [8, 90], [6, 90], [5, 90], [9, 120], [10, 120], [8, 120], [8, 120], [7, 120], [6, 120], [5, 120], [7, 120], [12, 180], [12, 180], [11, 180], [9, 180], [8, 180], [8, 180], [8, 180], [11, 180], [10, 180], [9, 180], [7, 180], [8, 180], [7, 180], [8, 180], [7, 180], [9, 180], [10, 180], [8, 180], [8, 180], [7, 180], [8, 180], [7, 180], [5, 180], [4, 180], [7, 180], [6, 180], [7, 180], [6, 180], [7, 180], [6, 180], [8, 180], [7, 180], [5, 180], [3, 180], [4, 180], [3, 180], [11, 360], [10, 360], [11, 360], [10, 360], [10, 360], [9, 360], [10, 360], [9, 360], [9, 360], [10, 360], [9, 360], [10, 360], [8, 360], [9, 360], [8, 360], [7, 360], [9, 360], [8, 360], [9, 360], [9, 360], [7, 360], [8, 360], [7, 360], [7, 360], [6, 360], [9, 360], [8, 360], [7, 360], [6, 360], [8, 360], [6, 360], [6, 360], [5, 360], [6, 360], [5, 360], [6, 360], [5, 360], [7, 360], [6, 360], [6, 360], [5, 360], [6, 360], [5, 360], [5, 360], [4, 360], [4, 360], [9, 720], [8, 720], [8, 720], [8, 720], [7, 720], [7, 720], [7, 720], [6, 720]]


def calc_cost(direc, inst, p, ep, pars, n, d):
    """
    Return the cost function at given meta parameters and parameters. Give pars as np.array.
    The path `inst` is the full path to the folder of the instance.
    """
    with open(inst + "/graph_input.txt", "r") as file:
        con = eval(file.readline())

    pars = ch.Variable(pars)

    if direc == "spatial":
        cost = _HQAOA.cost_from_parameters_swap_spatial(con, n, d, p, ep, pars)
    elif direc == "temporal":
        cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
            con, n, d, p, ep, pars
        )

    if type(cost) == ch.Variable:
        cost = cost.data

    return cost


def generate_landscape_data(direc_list, dirs, p_list, ep_list, randvecs, n, d):
    """
    Create the file landscape_data.txt, every line containing containing [direction,instance,p,ep,parameters,cost,parameters_id].
    Randvecs is a list of random parameter vectors, at each random vec the cost function is computed.
    The landscape_data.txt file is created for compatibility with a cluster workflow.
    """
    for direc in direc_list:
        for inst in dirs:
            for p in p_list:
                for ep in ep_list:
                    for par_i, pars in enumerate(randvecs):
                        cost = calc_cost(direc, inst, p, ep, pars, n, d)
                        output = str(
                            [direc, inst, p, ep, pars.tolist(), float(cost), par_i]
                        )
                        if not os.path.exists("landscape_data.txt"):
                            with open("landscape_data.txt", "a") as f:
                                f.write(
                                    "### The value of the cost function for various instances and points. Syntax [instance,direction,p,ep,parameters,cost,parameters_id]\n"
                                )
                        with open("landscape_data.txt", "a") as f:
                            f.write(output + "\n\n")
                            print(output)


def all_edge_data(graph):  # From a graph, return the a list of all edge weights.
    data = []
    edges = graph.edges()
    for edge in edges:
        data.append(graph.get_edge_data(*edge)["weight"])

    return data


def pm_instance_strings():
    """
    Convert the list of instances (stored as folder names) to +/- notation.
    They are returned as a numpy array containing the +/- weights.
    Outputs a list of tuples. First entry of tuple is the instance string in +/- notation, with ints in a list.
    Second is in 0/1 notation, as a string.
    """
    dirsbin = os.listdir("../data/SK/SWAP-network/6/spatial")
    dirsbin = [
        di for di in dirsbin if os.path.isdir("../data/SK/SWAP-network/6/spatial/" + di)
    ]
    dirsbin.sort()

    # Convert to ints and +- convention
    dirs = [list(di) for di in dirsbin]
    dirs = [list(map(int, di)) for di in dirs]
    dirs = np.array(dirs)
    dirs = -dirs * 2 + 1
    dirs = dirs.tolist()

    lst = [[dirs[i], dirsbin[i]] for i in range(len(dirs))]

    return lst


def get_class_number(
    classes, instance
):  # Get class index (as def by classes) of the graph with weights instance (in +/- notation).
    def em(e1, e2):  # Creterion for when two edges are considered equal.
        return e1["weight"] == e2["weight"]

    edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    edges = [[edge[0], edge[1], instance[i]] for i, edge in enumerate(edges)]
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge[0], edge[1], weight=edge[2])

    index = 0
    while not nx.is_isomorphic(g, classes[index][0], edge_match=em):
        index += 1

    return index


def import_landscape_data():
    """
    Import data from  landscape_data.txt.
    The latter is in the format with lines [direction,instance path,p,ep,parameters,cost,parameters_id].
    """
    with open("landscape_data.txt", "r") as f:
        f.readline()
        data = f.readlines()
        data = [eval(line.strip()) for line in data if line != "\n"]

    # Make list, every line gathers all data from one instance.
    path = "../data/SK/SWAP-network/6/spatial/"
    dirs = [
        path + inst for inst in sorted(os.listdir(path)) if os.path.isdir(path + inst)
    ]
    dirs.sort()
    per_inst = []
    for di in dirs:
        inst_data = [line for line in data if line[1] == di]
        per_inst.append(inst_data)
    return per_inst


def cost_vecs(p):
    """
    Gather all cost vectors that have a certain p.
    Every cost vector is a list of costs, each one calculated at one of the random points in parameter space.
    Every cost vector is from a different porblem instance.
    Note the cost vecs apear lexographical order of the instance bitstrings.
    """
    cost_vecs = []
    for inst in import_landscape_data():
        vec = [line[5] for line in inst if line[2] == p]
        cost_vecs.append(vec)

    l = len(cost_vecs[0])
    for vec in cost_vecs:
        assert len(vec) == l

    return np.array(cost_vecs)


def dists(p):
    """
    Compute the maximal absolute entry-wise distance between all pairs of cost vectors.
    Also compute average element-wise dist between all pairs.
    Return a list with entries [i,j,max_dist], where i,j both label an instance.
    """
    d = []
    cv = cost_vecs(p)
    for i in range(len(cv)):
        for j in range(i + 1, len(cv)):
            dif_vec = cv[i] - cv[j]
            max_dist = max(
                np.abs(dif_vec)
            )  # select the point with the max dist in abs val.
            d.append([i, j, max_dist])

    return d


def small_dists(p):
    """
    Make a list lines of lines [i,j] for which max_dist is smaller than 10**-14.
    Print the largest of those dists.
    """
    sd = [point for point in dists(p) if point[2] <= 10**-14]
    sd = np.array(sd, dtype=object)
    if len(sd) != 0:
        print(
            "Largest element-wise dist between the cost vecs of two instances in the same cost function class:"
        )
        print(max(sd.transpose()[2]))
        print()
    else:
        print("no small dists")
    return sd[:, :2]  # Only return the pairs, not the small dist.


def load_ground_space(
    instancebin,
):  # Load the ground space data of the instance. Instance should be given as string in 0/1 notation. Output is a tuple containing the ground space energy and the ground space.
    with open(
        "../data/SK/SWAP-network/6/spatial/" + str(instancebin) + "/ground_state.txt",
        "r",
    ) as f:
        a = f.readlines()
        gsp = [eval(line.strip()) for line in a]
        gspen = gsp[0][0]
        gsp = [ent[1] for ent in gsp]
        gsp = np.array(gsp)
        gsp = (-gsp + 1) / 2
        gsp = np.array(gsp, dtype=int)
        gsp = [str(line).replace(" ", "") for line in gsp]
        gsp = [str(line).replace("[", "") for line in gsp]
        gsp = [str(line).replace("]", "") for line in gsp]

    return gspen, gsp


if __name__ == "__main__":
    ## Generate and/or load data on graph classes.
    # compute_classes(6) # Uncomment to regenerate SK_graph_calsses_n=6.dat.
    with open("SK_graph_classes_n=6.dat", "rb") as file:
        classes = pickle.load(file)

    classes.sort(
        key=lambda x: x[2:0:-1]
    )  # Sort classes by multiplicity and then by ham weight.
    classes.reverse()  # Most common class goes first.
    print("On 6 qubits, there are", len(classes), "graph classes of the SK model.")

    ## Generate points for computing cost function classes
    instances = pm_instance_strings()
    n = 6
    d = 3
    path = "../data/SK/SWAP-network/6/spatial/"
    dirs = [path + instance[1] for instance in instances]
    n_random_points = 64
    randvecs = np.random.rand(n_random_points, 2 * d) * 100 - 50
    direc_list = ["spatial"]
    p_list = [0.0]
    ep_list = [0.0]
    # generate_landscape_data(direc_list,dirs,p_list,ep_list,randvecs,n,d) # Uncomment to regenerate ladscape_data.txt

    g = nx.Graph()
    g.add_edges_from(
        small_dists(0)[:, :2]
    )  # Construct graph where instances are nodes. There is an edge between them if the max element-wise dist of their cost vecs is below a threshold value. Then the connected components of this graph are the cost function classes.

    print(
        "The cost function landscape classes are", list(nx.connected_components(g)), ","
    )
    print(
        "where instances are represented by their lexcicographical order, and excluing landscape classes with a single member."
    )

    # Print info in the gap that divides the cost function landscape classes.
    large_dists = [point for point in dists(0) if point[2] > 10**-14]
    large_dists = np.array(large_dists, dtype=object)
    print(
        "Smallest element-wise dist between the cost vecs of two instances in different cost function classes:"
    )
    print(min(large_dists.transpose()[2]))
    print()

    # Gather all other data and print in table
    data = [
        ["Inst. bitstr.", "lex. ord.", "gr. cl. ind.", "gr. cl. mult.", "gs. en.", "gs"]
    ]
    cst_fn_cl = 0
    for i, instance in enumerate(instances):
        index = get_class_number(classes, instance[0])
        multiplicity = classes[index][2]
        ground_space = load_ground_space(instance[1])
        data.append([instance[1], i, index, multiplicity, *ground_space])
        cst_fn_cl += 1

    print(tabulate(data))

# Running this script gives stout:
# On 6 qubits, there are 156 graph classes of the SK model.
# Largest element-wise dist between the cost vecs of two instances in the same cost function class:
# 3.4416913763379853e-15
#
# The cost function landscape classes are [{1, 3, 5}, {9, 2, 12, 6}, {4, 7, 10, 11, 14, 15}] ,
# where instances are represented by their lexcicographical order, and excluing landscape classes with a single member.
# Smallest element-wise dist between the cost vecs of two instances in different cost function classes:
# 1.3977100408127525
#
# ---------------  ---------  ------------  -------------  -------  ------------------------------------------------------------------------------------------------------------------------
# Inst. bitstr.    lex. ord.  gr. cl. ind.  gr. cl. mult.  gs. en.  gs
# 000100100011110  0          42            360            -7.0     ['010101', '101010']
# 001100110101110  1          67            180            -9.0     ['001001', '110110']
# 001111011010001  2          2             720            -9.0     ['011011', '100100']
# 001111110000010  3          6             720            -9.0     ['011010', '100101']
# 010010100111110  4          1             720            -7.0     ['010000', '010110', '101001', '101111']
# 010011011101010  5          65            180            -9.0     ['010010', '101101']
# 010100101011100  6          36            360            -9.0     ['010101', '101010']
# 011100011010011  7          29            360            -7.0     ['010011', '010101', '101010', '101100']
# 011101001011100  8          69            180            -11.0    ['011001', '100110']
# 100010101000100  9          48            360            -9.0     ['001110', '110001']
# 100110010001101  10         4             720            -7.0     ['001001', '001100', '110011', '110110']
# 101100001010110  11         34            360            -7.0     ['001010', '010001', '101110', '110101']
# 110010010101000  12         7             720            -9.0     ['010010', '101101']
# 110101111100011  13         112           72             -5.0     ['000000', '000011', '000101', '001100', '010111', '011101', '100010', '101000', '110011', '111010', '111100', '111111']
# 111011011000010  14         1             720            -7.0     ['000010', '011010', '100101', '111101']
# 111100111011001  15         13            360            -7.0     ['000100', '001011', '110100', '111011']
# ---------------  ---------  ------------  -------------  -------  ------------------------------------------------------------------------------------------------------------------------
