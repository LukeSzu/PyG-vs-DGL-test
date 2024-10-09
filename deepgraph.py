import dgl
# Tworzenie prostego grafu
g = dgl.graph(([0, 1, 2], [1, 2, 3]))

# Wyświetlanie informacji o grafie
print("Liczba wierzchołków:", g.number_of_nodes())
print("Liczba krawędzi:", g.number_of_edges())
print("Indeksy krawędzi:", g.edges())