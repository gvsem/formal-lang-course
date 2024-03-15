from typing import Iterable
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
)
import numpy as np
from scipy.sparse import dok_matrix, kron, csr_array, csr_matrix, block_diag, vstack

from project.automata import graph_to_nfa, regex_to_dfa


def as_set(obj):
    if not isinstance(obj, set):
        return {obj}
    return obj


class FiniteAutomaton:

    m = None
    start = None
    final = None
    mapping = None

    def __init__(self, obj: any, start=set(), final=set(), mapping=dict()):
        if isinstance(obj, DeterministicFiniteAutomaton) or isinstance(
            obj, NondeterministicFiniteAutomaton
        ):
            mat = nfa_to_mat(obj)
            self.m, self.start, self.final, self.mapping = (
                mat.m,
                mat.start,
                mat.final,
                mat.mapping,
            )
        else:
            self.m, self.start, self.final, self.mapping = obj, start, final, mapping

    def accepts(self, word) -> bool:
        nfa = mat_to_nfa(self)
        real_word = "".join(list(word))
        return nfa.accepts(real_word)

    def is_empty(self) -> bool:
        return len(self.m) == 0 or len(list(self.m.values())[0]) == 0

    def size(self):
        return len(self.mapping)

    def mapping_for(self, u) -> int:
        return self.mapping[State(u)]

    def start_inds(self):
        return [self.mapping_for(t) for t in self.start]

    def final_inds(self):
        return [self.mapping_for(t) for t in self.final]

    def indexes_dict(self):
        return {i: v for v, i in self.mapping.items()}

    def labels(self):
        return self.m.keys()

    def matrix_of_direct_sum_with(self, other):
        m = dict()
        e = csr_matrix((other.size(), other.size()))
        for label in self.m.keys():
            if label in other.m.keys():
                m[label] = block_diag((self.m[label], other.m[label]), format="csr")
            else:
                m[label] = block_diag((self.m[label], e), format="csr")
        return m


def nfa_to_mat(automaton: NondeterministicFiniteAutomaton) -> FiniteAutomaton:
    states = automaton.to_dict()
    len_states = len(automaton.states)
    mapping = {v: i for i, v in enumerate(automaton.states)}
    m = dict()

    for label in automaton.symbols:
        m[label] = dok_matrix((len_states, len_states), dtype=bool)
        for u, edges in states.items():
            if label in edges:
                for v in as_set(edges[label]):
                    m[label][mapping[u], mapping[v]] = True

    return FiniteAutomaton(m, automaton.start_states, automaton.final_states, mapping)


def mat_to_nfa(automaton: FiniteAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for label in automaton.m.keys():
        m_size = automaton.m[label].shape[0]
        for u in range(m_size):
            for v in range(m_size):
                if automaton.m[label][u, v]:
                    nfa.add_transition(
                        automaton.mapping_for(u), label, automaton.mapping_for(v)
                    )

    for s in automaton.start:
        nfa.add_start_state(automaton.mapping_for(s))
    for s in automaton.final:
        nfa.add_final_state(automaton.mapping_for(s))

    return nfa


def transitive_closure(automaton: FiniteAutomaton):
    if len(automaton.m.values()) == 0:
        return dok_matrix((0, 0), dtype=bool)
    adjacency = sum(automaton.m.values())
    last_nnz = -1
    while adjacency.count_nonzero() != last_nnz:
        last_nnz = adjacency.count_nonzero()
        adjacency += adjacency @ adjacency

    return adjacency


def intersect_automata(
    automaton1: FiniteAutomaton, automaton2: FiniteAutomaton
) -> FiniteAutomaton:
    labels = automaton1.m.keys() & automaton2.m.keys()
    m = dict()
    start = set()
    final = set()
    mapping = dict()

    for label in labels:
        m[label] = kron(automaton1.m[label], automaton2.m[label], "csr")

    for u, i in automaton1.mapping.items():
        for v, j in automaton2.mapping.items():

            k = len(automaton2.mapping) * i + j
            mapping[State(k)] = k

            assert isinstance(u, State)
            if u in automaton1.start and v in automaton2.start:
                start.add(State(k))

            if u in automaton1.final and v in automaton2.final:
                final.add(State(k))

    return FiniteAutomaton(m, start, final, mapping)


def paths_ends(
    graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str
) -> list[tuple[object, object]]:

    graph_nfa = nfa_to_mat(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_dfa = nfa_to_mat(regex_to_dfa(regex))
    intersection = intersect_automata(graph_nfa, regex_dfa)
    closure = transitive_closure(intersection)

    mapping = {v: i for i, v in graph_nfa.mapping.items()}
    result = list()
    for u, v in zip(*closure.nonzero()):
        if u in intersection.start and v in intersection.final:
            result.append(
                (mapping[u // regex_dfa.size()], mapping[v // regex_dfa.size()])
            )
    return result


def reachability_with_constraints(
    fa: FiniteAutomaton, constraints_fa: FiniteAutomaton
) -> dict[int, set[int]]:

    m, n = constraints_fa.size(), fa.size()

    def get_front(s):
        front = dok_matrix((m, m + n), dtype=bool)
        for i in constraints_fa.start_inds():
            front[i, i] = True
        for i in range(m):
            front[i, s + m] = True
        return front

    def diagonalized(mat):
        result = dok_matrix(mat.shape, dtype=bool)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                if mat[j, i]:
                    result[i] += mat[j]
        return result

    labels = fa.labels() & constraints_fa.labels()
    result = {s: set() for s in fa.start}
    adj = {
        label: block_diag((constraints_fa.m[label], fa.m[label])) for label in labels
    }

    for v in fa.start_inds():
        front = get_front(v)
        for _ in range(m * n):
            front = sum(
                [dok_matrix((m, m + n), dtype=bool)]
                + [diagonalized(front @ adj[label]) for label in labels]
            )
            for i in constraints_fa.final_inds():
                for j in fa.final_inds():
                    if front[i, j + m]:
                        result[v].add(j)
    return result
