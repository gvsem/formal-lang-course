import pytest
from project.grammar import *
from pyformlang.cfg import CFG, Variable
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon


def test_grammar_weak1():

    actual = to_weak_normal_form(read_cfgrammar("tests/assets/cfg_test2"))
    expected = CFG.from_text(
        """
    S -> NP VP
    VP -> V NP
    V -> buys
    V -> touches
    NP -> georges
    NP -> jacques"""
    )

    assert actual.productions == expected.productions


def test_grammar_weak2():

    g = read_cfgrammar("tests/assets/cfg_test1")

    actual = to_weak_normal_form(g)

    expected = CFG(
        {Variable("S")},
        {Terminal("a"), Terminal("b")},
        Variable("S"),
        {
            Production(Variable("S"), [Epsilon()]),
            Production(Variable("S"), [Variable("a#CNF#"), Variable("C#CNF#1")]),
            Production(Variable("C#CNF#1"), [Variable("S"), Variable("b#CNF#")]),
            Production(Variable("a#CNF#"), [Variable("a")]),
            Production(Variable("b#CNF#"), [Variable("b")]),
        },
    )

    assert actual.productions == expected.productions
