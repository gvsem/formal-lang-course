from project.language.project.languageLexer import languageLexer
from project.language.project.languageParser import languageParser
from project.language.project.languageVisitor import languageVisitor
from project.language.project.languageListener import languageListener

from antlr4 import *
from antlr4.InputStream import InputStream


class CountVisitor(languageVisitor):

    count = 0

    def __init__(self):
        super(languageVisitor, self).__init__()

    def enterEveryRule(self, rule):
        self.count += 1

    def exitEveryRule(self, rule):
        pass


class SerializeVisitor(languageVisitor):

    res = ""

    def __init__(self):
        super(languageVisitor, self).__init__()

    def enterEveryRule(self, rule):
        res += rule.getText()

    def exitEveryRule(self, rule):
        pass


def get_parser(program: str) -> languageParser:
    return languageParser(CommonTokenStream(languageLexer(InputStream(program))))


def prog_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    parser = get_parser(program)
    ctx = parser.prog()
    return (ctx, parser.getNumberOfSyntaxErrors() == 0)


def nodes_count(tree: ParserRuleContext) -> int:
    visitor = CountVisitor()
    tree.accept(visitor)
    return visitor.count


def tree_to_prog(tree: ParserRuleContext) -> str:
    visitor = SerializeVisitor()
    tree.accept(visitor)
    return visitor.res
