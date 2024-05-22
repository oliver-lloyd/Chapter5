import networkx as nx


def molecule_to_graph(molecule):
    out = nx.Graph()
    for atom in molecule.GetAtoms():
        out.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    for bond in molecule.GetBonds():
        out.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
    return out


def MCS_proportion(graphs, i, j, outfile):
    g1 = graphs[i]
    g2 = graphs[j]
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        g1, g2,
        node_match=lambda n1, n2: n1['atomic_num'] == n2['atomic_num'],
        edge_match=lambda e1, e2: e1['bond_type'] == e2['bond_type']
    )
    largest_common_subgraph = max(matcher.subgraph_isomorphisms_iter(), key=len, default=None)
    out = [i, j]
    if largest_common_subgraph:
        proportion = len(largest_common_subgraph) / min(g1.number_of_nodes(), g2.number_of_nodes())
        out.append(proportion)
    else:
        out.append(0.0)
    with open(outfile, 'a') as f:
        f.write(','.join(map(str, out)) + '\n')
