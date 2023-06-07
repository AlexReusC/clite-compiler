# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import Literal, Variable, Visitor, BinaryOp, Declaration, Declarations, Assignment, Function, IfElse, Statement, Statements, Factor, WhileStatement, ForStatement, ReturnStatement, Program, FunctionCall
from llvmlite import ir

literals = ['+','-','*','/', '%', '(', ')', '{', '}', '<', '>', '=', ';', ',', '!']
reserved = {
    'else' : 'ELSE',
    'float' : 'FLOAT',
    'if' : 'IF',
    'int' : 'INT',
    'bool' : 'BOOL',
    'float' : 'FLOAT',
    'char' : 'CHAR',
    'return' : 'RETURN',
    'while' : 'WHILE',
    'for': 'FOR',
    'return': 'RETURN',
    'void': 'VOID'
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
    Program : Function Program
            | empty
    '''
    if len(p) > 2:
        p[0] = Program(p[1], p[2])

def p_Function(p):
    '''
    Function : FunctionReturnType ID '(' ')' '{' Declarations Statements ReturnStatement '}'
    '''
    p[0] = Function(p[1], p[2], p[6], p[7], p[8])

def p_ReturnStatement(p):
    '''
    ReturnStatement : RETURN Expression ';'
                    | RETURN ';'
    '''
    if len(p) > 3:
        p[0] = ReturnStatement(p[2])
    else:
        p[0] = ReturnStatement(False)

def p_empty(p):
    '''
    empty :
    '''
    pass

def p_Declarations(p):
    '''
    Declarations : Declaration Declarations
                 | empty
    '''
    if len(p) > 2:
        p[0] = Declarations(p[1], p[2])
    
def p_Declaration(p):
    '''
    Declaration : Type ID ';'
    '''
    p[0] = Declaration(p[2], p[1].upper())

def p_Type(p):
    '''
    Type : INT
         | BOOL
         | FLOAT
         | CHAR
    '''
    p[0] = p[1]

def p_FunctionReturnType(p):
    '''
    FunctionReturnType : Type
                       | VOID
    '''
    p[0] = p[1]

def p_Statements(p):
    '''
    Statements : Statement Statements
               | empty
    '''
    if len(p) > 2:
        p[0] = Statements(p[1], p[2])

def p_Statement(p):
    '''
    Statement : Assignment
              | IfStatement
              | WhileStatement
              | ForStatement
              | ';'
              | Block
    '''
    p[0] = p[1]

def p_Block(p):
    '''
    Block : '{' Statements '}'
    '''
    p[0] = p[2]

def p_IfStatement(p):
    '''
    IfStatement : IF '(' Expression ')' Statement ELSE Statement
    '''
    p[0] = IfElse(p[3], p[5], p[7])

def p_WhileStatement(p):
    '''
    WhileStatement : WHILE '(' Expression ')' Statement
    '''
    p[0] = WhileStatement(p[3], p[5])

def p_ForStatement(p):
    '''
    ForStatement : FOR '(' Assignment Expression ';' Assignment ')' Statement
    '''
    p[0] = ForStatement(p[3], p[4], p[6], p[8])

def p_Assignment(p):
    '''
    Assignment : ID '=' Expression ';'
    '''
    p[0] = Assignment(p[1], p[3])

def p_Expression(p):
    '''
    Expression : Conjunction
               | Expression OR Conjunction
    '''
    if len(p) > 2:
        p[0] = BinaryOp(p[2], p[1], p[3])
    else:
        p[0] = p[1]

def p_Conjunction(p):
    '''
    Conjunction : Equality
                | Conjunction AND Equality
    '''
    if len(p) > 2:
        p[0] = BinaryOp(p[2], p[1], p[3])
    else:
        p[0] = p[1]

def p_Equality(p):
    '''
    Equality : Relation
             | Relation EquOp Relation
    '''
    if len(p) > 2:
        p[0] = BinaryOp(p[2], p[1], p[3])
    else:
        p[0] = p[1]

def p_EquOp(p):
    """
    EquOp : EQ
          | NEQ
    """
    p[0] = p[1]

def p_Relation(p):
    '''
    Relation : Addition
             | Addition RelOp Addition 
    '''
    if len(p) > 2:
        p[0] = BinaryOp(p[2], p[1], p[3])
    else:
        p[0] = p[1]

def p_RelOp(p):
    """
    RelOp : '<'
          | LTE
          | '>'
          | GTE
    """
    p[0] = p[1]

def p_Addition(p):
    """
    Addition : Term
             | Addition AddOp Term
    """
    if len(p) > 2:
        p[0] = BinaryOp(p[2], p[1], p[3])
    else:
        p[0] = p[1]

def p_AddOp(p):
    """
    AddOp : '+'
          | '-'
    """
    p[0] = p[1]

def p_Term(p):
    '''
    Term : Factor
         | Term MulOp Primary
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])

def p_MulOp(p):
    """
    MulOp : '*'
          | '/'
          | '%'
    """
    p[0] = p[1]

def p_Factor(p):
    """
    Factor : Primary
           | UnaryOp Primary
    """
    #print("p", p[1], p[2])

    if len(p) > 2 and p[1] == '-':
        p[0] = p[2]
    elif len(p) > 2 and p[1] == '!':
        p[0] = p[2]
    else:
        p[0] = p[1]

def p_UnaryOp(p):
    """
    UnaryOp : '-'
            | '!'
    """
    p[0] = p[1]

def p_Primary_IntLit(p):
    'Primary : INTLIT'
    p[0] = Literal(p[1], 'INT')

def p_Primary_Id(p):
    'Primary : ID'
    p[0] = Variable(p[1], 'INT')

def p_Primary_FunctionCall(p):
    """
    Primary : ID '(' ')'
    """
    p[0] = FunctionCall(p[1])

# %%
intType = ir.IntType(32)
boolType = ir.IntType(32)
floatType = ir.FloatType()
charType = ir.IntType(32)
voidType = ir.VoidType()

class IRGenerator(Visitor):
    def __init__(self, module):
        self.stack = []
        self.symbolTable = dict()
        self.symbolFunc = dict()
        self.builder = None
        self.func = None
        self.module = module
        
    def visit_program(self, node: Program) -> None:
        node.function.accept(self)
        if node.program != None:
            node.program.accept(self)

    def visit_function(self, node: Function) -> None:
        return_type = None
        if node.functionReturnType == "int":
            return_type = intType
        else:
            return_type = voidType 

        fnty = ir.FunctionType(return_type, [])
        self.func = ir.Function(self.module, fnty, name=node.name)
        self.symbolFunc[node.name] = self.func
        entry = self.func.append_basic_block('entry')
        self.builder = ir.IRBuilder(entry)

        node.decls.accept(self)
        node.stats.accept(self)

        if node.functionReturnType != "void":
            node.returnStatement.accept(self)
        else:
            #self.stack.append(self.builder.ret_void())
            self.stack.append(self.builder.ret_void())


    def visit_return_statement(self, node: ReturnStatement) -> None:
        if node.expression: 
            node.expression.accept(self)
            tmp = self.stack.pop()
            self.builder.ret(tmp)


    def visit_if_else(self, node: IfElse) -> None:
        thenPart = self.func.append_basic_block('thenPart')
        elsePart = self.func.append_basic_block('elsePart')
        afterwards = self.func.append_basic_block('afterwards')
        node.expr.accept(self)
        expr = self.stack.pop()
        self.builder.cbranch(expr, thenPart, elsePart)

        # Then
        self.builder.position_at_start(thenPart)
        node.thenSt.accept(self)
        self.builder.branch(afterwards)

        # Else
        self.builder.position_at_start(elsePart)
        node.elseSt.accept(self)
        self.builder.branch(afterwards)

        self.builder.position_at_start(afterwards)

    def visit_while(self, node: WhileStatement) -> None:
        while_body = self.func.append_basic_block('whileBody')
        afterwards = self.func.append_basic_block('afterwards')

        node.expr.accept(self)
        expr = self.stack.pop()
        self.builder.cbranch(expr, while_body, afterwards)

        #body
        self.builder.position_at_start(while_body)
        node.st.accept(self)
        node.expr.accept(self)
        expr = self.stack.pop()
        self.builder.cbranch(expr, while_body, afterwards)

        #after
        self.builder.position_at_start(afterwards)

    def visit_for(self, node: ForStatement) -> None:
       """
       for(initial; expression; increment)
       statement
       afterwards 
       """

       expression_body = self.func.append_basic_block('expressionBody')
       increment_body = self.func.append_basic_block('incrementBody')
       statement_body = self.func.append_basic_block('statementBody') 
       afterwards = self.func.append_basic_block('afterwards')

       #initial
       node.initial.accept(self)
       #expresion
       node.expression.accept(self)
       expr = self.stack.pop()
       self.builder.cbranch(expr, statement_body, afterwards)
       #statement
       self.builder.position_at_start(statement_body)
       node.statement.accept(self)
       self.builder.branch(increment_body)
       #increment
       self.builder.position_at_start(increment_body)
       node.increment.accept(self)
       self.builder.branch(expression_body)
       #expresion
       self.builder.position_at_start(expression_body)
       node.expression.accept(self)
       expr = self.stack.pop()
       self.builder.cbranch(expr, statement_body, afterwards)
       #after
       self.builder.position_at_start(afterwards)


    def visit_declaration(self, node: Declaration) -> None:
        if node.type == 'INT':
            variable = self.builder.alloca(intType, name=node.name)
            self.symbolTable[node.name] = variable
        elif node.type == 'BOOL':
            variable = self.builder.alloca(boolType, name=node.name)
            self.symbolTable[node.name] = variable
        elif node.type == 'FLOAT':
            variable = self.builder.alloca(floatType, name=node.name)
            self.symbolTable[node.name] = variable
        elif node.type == 'CHAR':
            variable = self.builder.alloca(charType, name=node.name)
            self.symbolTable[node.name] = variable

    def visit_declarations(self, node: Declarations) -> None:
        node.declaration.accept(self)
        if node.declarations != None:
            node.declarations.accept(self)

    def visit_statement(self, node: Statement) -> None:
        pass

    def visit_statements(self, node: Statements) -> None:
        node.statement.accept(self)
        if node.statements != None:
            node.statements.accept(self)
        
    def visit_assignment(self, node: Assignment) -> None:
        node.rhs.accept(self)
        rhs = self.stack.pop()
        self.builder.store(rhs, self.symbolTable[node.lhs])

    def visit_literal(self, node: Literal) -> None:
        self.stack.append(intType(node.value))
    
    def visit_variable(self, node: Variable) -> None:
        self.stack.append(self.builder.load(self.symbolTable[node.name]))

    def visit_function_call(self, node: FunctionCall) -> None:
        self.stack.append(self.builder.call(self.symbolFunc[node.name], []))

    def visit_factor(self, node: Factor) -> None: #needs float

        if node.type == '!':
            #self.stack.append(self.builder.neg(node.value))
            pass
        elif node.type == '-':
            pass
            #self.stack.append(self.builder.neg(node.value))
        else:
            pass

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == '+':
            self.stack.append(self.builder.add(lhs, rhs))
        elif node.op == '-':
            self.stack.append(self.builder.sub(lhs, rhs))
        elif node.op == '*':
            self.stack.append(self.builder.mul(lhs, rhs))
        elif node.op == '%':
            self.stack.append(self.builder.srem(lhs, rhs))
        elif node.op in ['<', '>', '>=', '<=', '==', '!=']:
            self.stack.append(
                    self.builder.icmp_signed(node.op, lhs, rhs),
            )
        elif node.op == '&&':
            self.stack.append(self.builder.and_(lhs, rhs))
        elif node.op == '||':
            self.stack.append(self.builder.or_(lhs, rhs))

module = ir.Module(name="prog")

data =  '''
        int fact() {
            int x;
            int i;

            x = 1;
            i = 1;
            while(i <= 5){
                x = x * i;
                i = i + 1;
            }
            return x;
        }

        void voidFunction() {
            int x;
            x = 1;
            return;
        }


        int main() {
            int x;
            int y;
            bool t;
            int i;

            t = 1;

            if (1 == 1 || 1 == 1 )
                x = 1;
            else
                x = 1;
            x = 1;
            for(i = 0 ; 1 > 1; i = i + 1;){
                x = 2;
                x = 1;
            }

            y = fact();

            return y;
        }
        '''
lexer = lex.lex()
parser = yacc.yacc()
ast = parser.parse(data)
print(ast)

visitor = IRGenerator(module)
ast.accept(visitor)
# builder.ret(visitor.stack.pop())

print(module)



import runtime as rt
from ctypes import CFUNCTYPE, c_int

engine = rt.create_execution_engine()
mod = rt.compile_ir(engine, str(module))
func_ptr = engine.get_function_address("main")

cfunc = CFUNCTYPE(c_int)(func_ptr)
res = cfunc()
print(res)



# %%