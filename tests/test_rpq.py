import networkx
from networkx import MultiDiGraph
from project.automaton import paths_ends
from cfpq_data import *

from project.graphutils import save_twocycled_graph


def test_path_ends_1():

    g = MultiDiGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1, label="a")

    result = paths_ends(g, {0}, {1}, "a")
    assert result == [(0, 1)]


def test_path_ends_3():

    g = MultiDiGraph()
    g.add_node(0)
    g.add_edge(0, 0, label="a")

    result = paths_ends(g, {0}, {0}, "a*")
    assert result == [(0, 0)]
