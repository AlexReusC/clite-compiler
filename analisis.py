# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import Literal

literals = ['+','-','*','/', '%', '(', ')']
tokens = ['ID', 'INTLIT']

t_ignore  = ' \t'

def t_ID(t):
     r'[a-zA-Z_][a-zA-Z_0-9]*'
     return t

def t_INTLIT(t):
    r'[0-9]+'
    t.value = int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# %%
def p_Term(p):
    '''
    Term : Term '*' Primary
         | Primary
    '''
    if len(p) == 4:
        p[0] = BinaryOp("*", p[1], p[3])
    else:
        p[0] = p[1]

def p_Primary(p):
    '''
    Primary : INTLIT 
            | '(' Primary ')'
    '''
    if len(p) == 2:
        p[0] = Literal(p[1], 'INT')
    else:
        p[0] = p[2]
        
def p_error(p):
    print("Syntax error in input!", p)




#%%
from arbol import Literal, BinaryOp, Calculator, Visitor, Variable
from llvmlite import ir

intType = ir.IntType(32)


class IRGenerator(Visitor):
    def init(self, builder):
        self.stack = []
        self.builder = builder

    def visit_literal(self, node: Literal) -> None:
        self.stack.append(intType(node.value))
   
    def visit_variable(self, node: Variable) -> None:
        pass

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop() # estos van a ser ints
        lhs = self.stack.pop() # estos van a ser ints

        if node.op == '+':
            self.stack.append(self.builder.add(lhs, rhs))
        if node.op == '*':
            self.stack.append(self.builder.mul(lhs, rhs))
        if node.op == '-':
            self.stack.append(self.builder.sub(lhs, rhs))
        
            #div y srem
            
module = ir.Module(name="prog")

# int main() {
fnty = ir.FunctionType(intType, [])
func = ir.Function(module, fnty, name='main')

entry = func.append_basic_block('entry')
builder = ir.IRBuilder(entry)

#####
data = '10 * 5'
lexer = lex.lex()
parser = yacc.yacc()

uno = parser.parse(data)
calc = IRGenerator(builder)

uno.accept(calc)
temp = calc.stack.pop()
#####

# builder.add(lhs, rhs)
builder.ret(temp)
# builder = calc.builder
print(module)