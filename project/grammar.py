import pyformlang
import scipy
from project import automaton
from pyformlang.cfg import CFG, Variable, Terminal, Epsilon
from typing import Tuple
from scipy.sparse import dok_matrix, csr_matrix
from networkx import DiGraph
from pyformlang.regular_expression import Regex
from project.automata import graph_to_nfa, regex_to_dfa
from pyformlang.rsa.box import Box
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import Symbol
from pyformlang.finite_automaton.finite_automaton import to_symbol


def read_cfgrammar(path, start="S") -> CFG:
    with open(path, "r") as f:
        text = f.read()
        return CFG.from_text(text, Variable(start))


def to_weak_normal_form(grammar, start="S"):
    elim_cfg = grammar.eliminate_unit_productions().remove_useless_symbols()
    productions = elim_cfg._decompose_productions(
        elim_cfg._get_productions_with_only_single_terminals()
    )
    return CFG(productions=set(productions), start_symbol=Variable(start))


def cfg_to_weak_normal_form(grammar):
    return to_weak_normal_form(grammar)


def cfpq_with_hellings(
    cfg,
    graph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[Tuple[int, int]]:

    # Weak Homsky Normal Form -
    # 1. N_i -> t_j
    # 2. N_i -> eps
    # 3. N_i -> N_j N_k

    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    g = to_weak_normal_form(cfg)
    p1 = {}
    p2 = set()
    p3 = {}

    for p in g.productions:
        if len(p.body) == 1 and isinstance(p.body[0], Terminal):
            p1.setdefault(p.head, set()).add(p.body[0])
        elif len(p.body) == 1 and isinstance(p.body[0], Epsilon):
            p2.add(p.body[0])
        elif len(p.body) == 2:
            p3.setdefault(p.head, set()).add((p.body[0], p.body[1]))

    r = {(N_i, v, v) for N_i in p2 for v in graph.nodes}
    r |= {
        (N_i, v, u)
        for (v, u, tag) in graph.edges.data("label")
        for N_i in p1
        if tag in p1[N_i]
    }

    m = r.copy()

    while len(m) > 0:
        N_i, v, u = m.pop()

        r_tmp = set()
        for N_j, v_, u_ in r:
            if v == u_:
                for N_k in p3:
                    if (N_j, N_i) in p3[N_k] and (N_k, v_, v) not in r:
                        m.add((N_k, v_, u))
                        r_tmp.add((N_k, v_, u))

        for N_j, v_, u_ in r:
            if u == v_:
                for N_k in p3:
                    if (N_i, N_j) in p3[N_k] and (N_k, v, u_) not in r:
                        m.add((N_k, v, u_))
                        r_tmp.add((N_k, v, u_))

        r |= r_tmp

    return {
        (v, u)
        for (N_i, v, u) in r
        if v in start_nodes and u in final_nodes and Variable(N_i) == cfg.start_symbol
    }


def cfpq_with_matrix(
    cfg,
    graph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[Tuple[int, int]]:

    cfg = to_weak_normal_form(cfg)
    n = len(graph.nodes)
    E = graph.edges.data("label")
    T = {var: dok_matrix((n, n), dtype=bool) for var in cfg.variables}

    p3 = set()  # N_i -> N_j N_k

    for i, j, tag in E:
        for p in cfg.productions:
            if (
                len(p.body) == 1
                and isinstance(p.body[0], Variable)
                and p.body[0].value == tag
            ):
                T[p.head][i, j] = True
            elif len(p.body) == 1 and isinstance(p.body[0], Epsilon):
                T[p.head] += csr_matrix(scipy.eye(n), dtype=bool)
            elif len(p.body) == 2:
                p3.add((p.head, p.body[0], p.body[1]))

    r = {i: node for i, node in enumerate(graph.nodes)}
    T = {x: csr_matrix(m) for (x, m) in T.items()}

    changed = True
    while changed:
        changed = False
        for N_i, N_j, N_k in p3:
            prev = T[N_i].nnz
            T[N_i] += T[N_j] @ T[N_k]
            changed |= prev != T[N_i].nnz

    return {(r[i], r[j]) for _, m in T.items() for i, j in zip(*m.nonzero())}


def cfpq_with_tensor(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:

    mat = automaton.rsm_to_mat(rsm)
    graph_mat = automaton.nfa_to_mat(graph_to_nfa(graph, start_nodes, final_nodes))
    mat_inds = mat.indexes_dict()
    graph_mat_inds = graph_mat.indexes_dict()

    n = graph_mat.states_count

    for var in mat.nullable_symbols:
        if var not in graph_mat.m:
            graph_mat.m[var] = dok_matrix((n, n), dtype=bool)
        graph_mat.m[var] += scipy.sparse.eye(n, dtype=bool)

    last_nnz = 0
    while True:

        closure = automaton.transitive_closure(
            automaton.intersect_automata(mat, graph_mat)
        ).nonzero()
        closure = list(zip(*closure))

        curr_nnz = len(closure)
        if curr_nnz == last_nnz:
            break
        last_nnz = curr_nnz

        for i, j in closure:
            src = mat_inds[i // n]
            dst = mat_inds[j // n]

            if src in mat.start and dst in mat.final:
                var = src.value[0]
                if var not in graph_mat.m:
                    graph_mat.m[var] = dok_matrix((n, n), dtype=bool)
                graph_mat.m[var][i % n, j % n] = True

    return {
        (graph_mat_inds[i], graph_mat_inds[j])
        for _, m in graph_mat.m.items()
        for i, j in zip(*m.nonzero())
        if graph_mat_inds[i] in mat.start and graph_mat_inds[j] in mat.final
    }


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    prods = {}
    for p in cfg.productions:
        if len(p.body) == 0:
            regex = Regex(
                " ".join(
                    "$" if isinstance(var, Epsilon) else var.value for var in p.body
                )
            )
        else:
            regex = Regex("$")
        if Symbol(p.head) not in prods:
            prods[Symbol(p.head)] = regex
        else:
            prods[Symbol(p.head)] = prods[Symbol(p.head)].union(regex)

    prods = {
        Symbol(var): Box(regex.to_epsilon_nfa().to_deterministic(), Symbol(var))
        for var, regex in prods.items()
    }

    return pyformlang.rsa.RecursiveAutomaton(
        set(prods.keys()), Symbol("S"), set(prods.values())
    )


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    prods = {}
    boxes = set()
    for p in ebnf.splitlines():
        p = p.strip()
        if "->" not in p:
            continue

        head, body = p.split("->")
        head = head.strip()
        body = body.strip() if body.strip() != "" else Epsilon().to_text()

        if head in prods:
            prods[head] += " | " + body
        else:
            prods[head] = body

    prods = {
        Symbol(var): Box(Regex(regex).to_epsilon_nfa().to_deterministic(), Symbol(var))
        for var, regex in prods.items()
    }

    return RecursiveAutomaton(set(prods.keys()), Symbol("S"), set(prods.values()))
