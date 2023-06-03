from arbol import Literal, BinaryOp, Calculator, Visitor, Variable, ASTNode
from llvmlite import ir

intType = ir.IntType(32)

class IRGenerator(Visitor):
    def __init__(self, builder):
        self.builder = builder
        self.stack = []

    def visit_literal(self, node: Literal) -> None:
        self.stack.append(intType(node.value))


    def visit_binary_op(self, node: BinaryOp) -> None:
        print("hola")
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == '+':
            self.stack.append(self.builder.add(lhs, rhs))
        elif node.op == '-':
            self.stack.append(lhs - rhs)
        elif node.op == '*':
            self.stack.append(lhs * rhs)
        elif node.op == '/':
            self.stack.append(lhs / rhs)
        elif node.op == '%':
            self.stack.append(lhs % rhs)
   
    def visit_variable(self, node: Variable) -> None:
        pass


module = ir.Module(name="prog")

# int main() {
fnty = ir.FunctionType(intType, [])
func = ir.Function(module, fnty, name='main')

entry = func.append_basic_block('entry')
builder = ir.IRBuilder(entry)

#####
uno = BinaryOp('+', Literal(10, "INT"), Literal(12, "INT"))
#uno = Literal(10, "INT")
calc = IRGenerator(builder)

uno.accept(calc)
temp = calc.stack.pop()

#####

builder.ret(temp)

print(module)