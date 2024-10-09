import time
import dgl
from dgl.nn import GraphConv
import torch
from torch_geometric.nn import GCNConv
import torch_geometric.data
import networkx as nx
import matplotlib.pyplot as plt

data_nodes_direct = []
data_degrees_direct = []
data_nodes_indirect = []
data_degrees_indirect = []

# Funkcja do pomiaru czasu wykonania message passing
def measure_time_passing(graph, conv_layer, features):
    times = []
    for _ in range(10):
        start_time = time.time()
        _ = conv_layer(graph, features)
        end_time = time.time()
        times.append((end_time - start_time)*1000)
    return sum(times) / len(times)

def pass_message(nodes,degree, type):

    ba_graph = nx.barabasi_albert_graph(nodes, degree, seed=2137)
    if type == "direct":
        ba_graph = ba_graph.to_directed()
        
    edges = list(ba_graph.edges())

    #PYG
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x_pyg = torch.randn(nodes, 16)  # n to liczba wierzchołków w twoim grafie
    data_pyg = torch_geometric.data.Data(x=x_pyg, edge_index=edge_index)
    # Utwórz i zainicjalizuj warstwę przekazywania wiadomości
    conv_layer_pyg = GCNConv(16, 16)

    avg_time_pyg = measure_time_passing(data_pyg.x, conv_layer_pyg, data_pyg.edge_index)

    #DGL
    g_dgl = dgl.from_networkx(ba_graph)

    # Inicjalizuj cechy wierzchołków
    g_dgl.ndata['feat'] = torch.randn(g_dgl.number_of_nodes(), 16)

    # Utwórz i zainicjalizuj warstwę przekazywania wiadomości
    conv_layer_dgl = GraphConv(16, 16)  # Warstwa przekazywania wiadomości DGL

    avg_time_dgl = measure_time_passing(g_dgl, conv_layer_dgl, g_dgl.ndata['feat'])
    new_record = {"nodes": nodes, "degree": degree, "PyG": avg_time_pyg, "DGL": avg_time_dgl, "type": type}
    return new_record



for node in range(1000, 20001, 500):
    data_nodes_direct.append(pass_message(node,10, "direct"))
    data_nodes_indirect.append(pass_message(node,10, "indirect"))
    print(node, "/", 20000)


nodes_values = []
avg_time_pyg_values_direct = []
avg_time_dgl_values_direct = []
avg_time_pyg_values_indirect = []
avg_time_dgl_values_indirect = []

for entry in data_nodes_direct:
    nodes_values.append(entry["nodes"])
    avg_time_pyg_values_direct.append(entry["PyG"])
    avg_time_dgl_values_direct.append(entry["DGL"])

for entry in data_nodes_indirect:
    avg_time_pyg_values_indirect.append(entry["PyG"])
    avg_time_dgl_values_indirect.append(entry["DGL"])

average_direct_pyg = sum(avg_time_pyg_values_direct)/len(avg_time_pyg_values_direct)
average_direct_dgl = sum(avg_time_dgl_values_direct)/len(avg_time_dgl_values_direct)
average_indirect_pyg = sum(avg_time_pyg_values_indirect)/len(avg_time_pyg_values_indirect)
average_indirect_dgl = sum(avg_time_dgl_values_indirect)/len(avg_time_dgl_values_indirect)
data = [["graf", "PyG", "DGL"],["skierowany", round(average_direct_pyg,3), round(average_direct_dgl,3)],["nieskierowany", round(average_indirect_pyg,3), round(average_indirect_dgl,3)]]

plt.clf()
plt.figure()
plt.plot(nodes_values, avg_time_pyg_values_direct, label="PyG direct")
plt.plot(nodes_values, avg_time_dgl_values_direct, label="DGL direct")
plt.plot(nodes_values, avg_time_pyg_values_indirect, label="PyG indirect")
plt.plot(nodes_values, avg_time_dgl_values_indirect, label="DGL indirect")
plt.xticks(range(min(nodes_values), max(nodes_values)+1, 2000), rotation=45)
plt.xlabel("Nodes")
plt.ylabel("Time [ms]")
plt.title("Comparison of PyG and DGL")
plt.legend()
plt.grid(True)
plt.savefig('D:/badawcze/1.png')

for degree in range(10, 101, 5):
    data_degrees_direct.append(pass_message(10000,degree, "direct"))
    data_degrees_indirect.append(pass_message(10000,degree, "indirect"))
    print(degree, "/" ,100)

degree_values = []
avg_time_pyg_values_direct = []
avg_time_dgl_values_direct = []
avg_time_pyg_values_indirect = []
avg_time_dgl_values_indirect = []

for entry in data_degrees_direct:
    degree_values.append(entry["degree"])
    avg_time_pyg_values_direct.append(entry["PyG"])
    avg_time_dgl_values_direct.append(entry["DGL"])

for entry in data_degrees_indirect:
    avg_time_pyg_values_indirect.append(entry["PyG"])
    avg_time_dgl_values_indirect.append(entry["DGL"])

plt.clf()
plt.figure()

plt.plot(degree_values, avg_time_pyg_values_direct, label="PyG direct")
plt.plot(degree_values, avg_time_dgl_values_direct, label="DGL direct")
plt.plot(degree_values, avg_time_pyg_values_indirect, label="PyG indirect")
plt.plot(degree_values, avg_time_dgl_values_indirect, label="DGL indirect")
plt.xticks(range(min(degree_values), max(degree_values)+1, 10), rotation=45)
plt.xlabel("Degrees")
plt.ylabel("Time [ms]")
plt.title("Comparison of PyG and DGL")
plt.legend()
plt.grid(True)
plt.savefig('D:/badawcze/2.png')


plt.clf()
plt.figure()
fig, ax = plt.subplots()
table = ax.table(cellText=data, loc='center')
ax.axis('off')
plt.savefig('D:/badawcze/3.png')