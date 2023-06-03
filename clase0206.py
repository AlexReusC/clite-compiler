# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import Literal, Variable, Visitor, BinaryOp, Declaration, Declarations, Assignment, Program
from llvmlite import ir

literals = ['+','-','*','/', '%', '(', ')', '{', '}', '<', '>', '=', ';', ',']
reserved = {
    'else' : 'ELSE',
    'float' : 'FLOAT',
    'if' : 'IF',
    'int' : 'INT',
    'main' : 'MAIN',
    'return' : 'RETURN',
    'while' : 'WHILE'
}

tokens = list(reserved.values()) + ['ID', 'INTLIT', 'LTE', 'GTE', 'EQ', 'NEQ', 'AND', 'OR']

t_ignore  = ' \t'

t_LTE = r'<='
t_GTE = r'>='
t_EQ = r'=='
t_NEQ = r'!='
t_AND = r'&&'
t_OR = r'\|\|'

def t_ID(t):
     r'[a-zA-Z_][a-zA-Z_0-9]*'
     t.type = reserved.get(t.value,'ID')    # Check for reserved words
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


# ========================
def p_Program(p):
    '''
    Program : INT MAIN '(' ')' '{' Declarations Statements '}'
    '''
    p[0] = Program( p[6], p[7] )

# def p_empty(p):
#     '''
#     empty :
#     '''
#     pass

def p_Declarations(p):
    '''
    Declarations : Declaration
    '''
    p[0] = p[1]
    # '''
    # Declarations : Declaration Declarations
    #              | empty
    # '''
    # if len(p) > 2:
    #     p[0] = Declarations(p[1], p[2])
    
def p_Declaration(p):
    '''
    Declaration : INT ID ';'
    '''
    p[0] = Declaration(p[2], p[1].upper())

def p_Statements(p):
    '''
    Statements : Statement
    '''
    p[0] = p[1]

def p_Statement(p):
    '''
    Statement : Assignment
    '''
    p[0] = p[1]

def p_Assignment(p):
    '''
    Assignment : ID '=' Expression ';'
    '''
    p[0] = Assignment(p[1], p[3])

def p_Expression(p):
    '''
    Expression : Conjunction
    '''
    p[0] = p[1]

def p_Conjunction(p):
    '''
    Conjunction : Equality
    '''
    p[0] = p[1]

def p_Equality(p):
    '''
    Equality : Relation
    '''
    p[0] = p[1]

def p_Relation(p):
    '''
    Relation : Addition
             | Relation '<' Addition 
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])

def p_Addition(p):
    '''
    Addition : Factor
    '''
    p[0] = p[1]

def p_Factor(p):
    '''
    Factor : Primary
           | Factor '*' Primary
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])

def p_Primary_IntLit(p):
    'Primary : INTLIT'
    p[0] = Literal(p[1], 'INT')

def p_Primary_Id(p):
    'Primary : ID'
    p[0] = Variable(p[1], 'INT')

# %%
intType = ir.IntType(32)

class IRGenerator(Visitor):
    def __init__(self, builder):
        self.stack = []
        self.symbolTable = dict()
        self.builder = builder

    def visit_program(self, node: Program) -> None:
        node.decls.accept(self)
        node.stats.accept(self)

    def visit_declaration(self, node: Declaration) -> None:
        if node.type == 'INT':
            variable = self.builder.alloca(intType, name=node.name)
            self.symbolTable[node.name] = variable
        
    def visit_assignment(self, node: Assignment) -> None:
        node.rhs.accept(self)
        rhs = self.stack.pop()
        self.builder.store(rhs, self.symbolTable[node.lhs])

    def visit_literal(self, node: Literal) -> None:
        self.stack.append(intType(node.value))
    
    def visit_variable(self, node: Variable) -> None:
        self.stack.append(self.builder.load(self.symbolTable[node.name]))

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == '+':
            self.stack.append(self.builder.add(lhs, rhs))
        elif node.op == '*':
            self.stack.append(self.builder.mul(lhs, rhs))

module = ir.Module(name="prog")

fnty = ir.FunctionType(intType, [])
func = ir.Function(module, fnty, name='main')

entry = func.append_basic_block('entry')
builder = ir.IRBuilder(entry)

data =  '''
        int main() {
            int x;
            
            x = 5;
        }
        '''
lexer = lex.lex()
parser = yacc.yacc()
ast = parser.parse(data)
print(ast)

visitor = IRGenerator(builder)
ast.accept(visitor)
# builder.ret(visitor.stack.pop())

print(module)

# %%