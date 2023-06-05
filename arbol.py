from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any



class ASTNode(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass

class Program(ASTNode):
    def __init__(self, decls: Any, stats: Any) -> None:
        self.decls = decls
        self.stats = stats

    def accept(self, visitor: Visitor):
        visitor.visit_program(self)

class Literal(ASTNode):
    def __init__(self, value: Any, type: str) -> None:
        self.value = value
        self.type = type

    def accept(self, visitor: Visitor):
        visitor.visit_literal(self)

class Variable(ASTNode):
    def __init__(self, name: Any, type: str) -> None:
        self.name = name
        self.type = type

    def accept(self, visitor: Visitor):
        visitor.visit_variable(self)


class IfElse(ASTNode):
    def __init__(self, expr: Any, thenSt: Any, elseSt: Any) -> None:
        self.expr = expr
        self.thenSt = thenSt
        self.elseSt = elseSt
    
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_if_else(self)

class WhileStatement(ASTNode):
    def __init__(self, expr: Any, st: Any) -> None:
        self.expr = expr
        self.st = st

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_while(self)

class ForStatement(ASTNode):
    def __init__(self, initial: Assignment, expression: Any, increment: Assignment, statement: Statement) -> None:
        self.initial = initial 
        self.expression = expression
        self.increment = increment 
        self.statement = statement

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_for(self)

class BinaryOp(ASTNode):
    def __init__(self, op: str, lhs: ASTNode, rhs: ASTNode) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def accept(self, visitor: Visitor):
        visitor.visit_binary_op(self)

class Factor(ASTNode):
    def __init__(self, value: str, type: str) -> None:
        self.value = value
        self.type = type

    def accept(self, visitor: Visitor):
        visitor.visit_factor(self)

class Declaration(ASTNode):
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type

    def accept(self, visitor: Visitor):
        visitor.visit_declaration(self)

class Declarations(ASTNode):
    def __init__(self, declaration: Declaration, declarations: Declarations) -> None:
        self.declaration = declaration
        self.declarations = declarations

    def accept(self, visitor: Visitor):
        visitor.visit_declarations(self)

class Statement(ASTNode):
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_statement(self)

class Statements(ASTNode):
    def __init__(self, statement: Statement, statements: Statements) -> None:
        self.statement = statement
        self.statements = statements

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_statements(self)

class Assignment(ASTNode):
    def __init__(self, lhs: str, rhs: ASTNode) -> None:
        self.lhs = lhs
        self.rhs = rhs
    
    def accept(self, visitor: Visitor):
        visitor.visit_assignment(self)

class Visitor(ABC):
    @abstractmethod
    def visit_literal(self, node: Literal) -> None:
        pass
    @abstractmethod
    def visit_variable(self, node: Variable) -> None:
        pass
    @abstractmethod
    def visit_binary_op(self, node: BinaryOp) -> None:
        pass

class Calculator(Visitor):
    def __init__(self):
        self.stack = []

    def visit_literal(self, node: Literal) -> None:
        self.stack.append(node.value)
    
    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == '+':
            self.stack.append(lhs + rhs)
        elif node.op == '-':
            self.stack.append(lhs - rhs)
        elif node.op == '*':
            self.stack.append(lhs * rhs)
        elif node.op == '/':
            self.stack.append(lhs / rhs)
        elif node.op == '%':
            self.stack.append(lhs % rhs)