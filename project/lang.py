from project.language.project.languageLexer import languageLexer
from project.language.project.languageParser import languageParser
from project.language.project.languageVisitor import languageVisitor

from antlr4 import *
from antlr4.InputStream import InputStream


class CountVisitor(languageVisitor):

    ctx = None
    count = 0

    def __init__(self, ctx):
        super(languageVisitor, self).__init__()
        self.ctx = ctx

    def visit(self):
        ParseTreeWalker().walk(self, self.ctx)

    def enterEveryRule(self, rule):
        self.count += 1

    def exitEveryRule(self, rule):
        pass


class SerializeVisitor(languageVisitor):

    ctx = None
    res = ""

    def __init__(self, ctx):
        super(languageVisitor, self).__init__()
        self.ctx = ctx

    def visit(self):
        ParseTreeWalker().walk(self, self.ctx)

    def enterEveryRule(self, rule):
        res += self.rules[rule.getRuleIndex()]

    def exitEveryRule(self, rule):
        pass


def get_parser(program: str) -> languageParser:
    return languageParser(CommonTokenStream(languageLexer(InputStream(program))))


def prog_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    parser = get_parser(program)
    ctx = parser.prog()
    return (ctx, parser.getNumberOfSyntaxErrors() == 0)


def nodes_count(tree: ParserRuleContext) -> int:
    return CountVisitor(tree).count


def tree_to_prog(tree: ParserRuleContext) -> str:
    return SerializeVisitor(tree).res
