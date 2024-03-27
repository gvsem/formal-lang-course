from pyformlang.cfg import CFG, Variable


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
