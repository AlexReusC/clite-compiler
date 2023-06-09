
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = "AND BOOL BOOLLIT CHAR ELSE EQ FLOAT FLOATLIT FOR GTE ID IF INT INTLIT LTE NEQ OR RETURN VOID WHILE\n    Program : Function Program\n            | empty\n    \n    Function : FunctionReturnType ID '(' ')' '{' Declarations Statements ReturnStatement '}'\n             | FunctionReturnType ID '(' Params ')' '{' Declarations Statements ReturnStatement '}'\n    \n    Params : Param Params\n           | empty\n    \n    Param : Type ID\n    \n    ReturnStatement : RETURN Expression ';'\n                    | RETURN ';'\n    \n    empty :\n    \n    Declarations : Declaration Declarations\n                 | empty\n    \n    Declaration : Type ID ';'\n    \n    Type : INT\n         | BOOL\n         | FLOAT\n         | CHAR\n    \n    FunctionReturnType : Type\n                       | VOID\n    \n    Statements : Statement Statements\n               | empty\n    \n    Statement : Assignment\n              | IfStatement\n              | WhileStatement\n              | ForStatement\n              | ';'\n              | Block\n    \n    Block : '{' Statements '}'\n    \n    IfStatement : IF '(' Expression ')' Statement ELSE Statement\n    \n    WhileStatement : WHILE '(' Expression ')' Statement\n    \n    ForStatement : FOR '(' Assignment Expression ';' Assignment ')' Statement\n    \n    Assignment : ID '=' Expression ';'\n    \n    Expression : Conjunction\n               | Expression OR Conjunction\n    \n    Conjunction : Equality\n                | Conjunction AND Equality\n    \n    Equality : Relation\n             | Relation EquOp Relation\n    \n    EquOp : EQ\n          | NEQ\n    \n    Relation : Addition\n             | Addition RelOp Addition \n    \n    RelOp : '<'\n          | LTE\n          | '>'\n          | GTE\n    \n    Addition : Term\n             | Addition AddOp Term\n    \n    AddOp : '+'\n          | '-'\n    \n    Term : Factor\n         | Term MulOp Primary\n    \n    MulOp : '*'\n          | '/'\n          | '%'\n    \n    Factor : Primary\n           | UnaryOp Primary\n    \n    UnaryOp : '-'\n            | '!'\n    Primary : INTLITPrimary : FLOATLITPrimary : BOOLLITPrimary : ID\n    Primary : ID '(' ')'\n            | ID '(' Primary ')'\n    "
    
_lr_action_items = {'$end':([0,1,2,3,11,71,102,],[-10,0,-10,-2,-1,-3,-4,]),'VOID':([0,2,71,102,],[6,6,-3,-4,]),'INT':([0,2,13,16,19,22,24,27,53,71,102,],[7,7,7,7,7,-7,7,7,-13,-3,-4,]),'BOOL':([0,2,13,16,19,22,24,27,53,71,102,],[8,8,8,8,8,-7,8,8,-13,-3,-4,]),'FLOAT':([0,2,13,16,19,22,24,27,53,71,102,],[9,9,9,9,9,-7,9,9,-13,-3,-4,]),'CHAR':([0,2,13,16,19,22,24,27,53,71,102,],[10,10,10,10,10,-7,10,10,-13,-3,-4,]),'ID':([4,5,6,7,8,9,10,18,19,23,24,25,26,27,29,31,33,34,35,36,37,38,42,44,45,48,50,51,52,53,64,68,69,70,76,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,99,100,112,113,115,117,118,119,],[12,-18,-19,-14,-15,-16,-17,22,-10,28,-10,-12,43,-10,28,28,-22,-23,-24,-25,-26,-27,-11,28,55,55,55,55,28,-13,55,-58,-59,-28,55,55,-32,55,55,55,-39,-40,55,55,-43,-44,-45,-46,-49,-50,55,-53,-54,-55,28,28,-30,28,28,-29,28,-31,]),'(':([12,39,40,41,55,],[13,50,51,52,78,]),')':([13,15,16,17,21,22,55,57,58,59,60,61,62,63,65,66,67,74,75,78,79,97,103,104,105,106,107,108,109,110,114,116,],[14,20,-10,-6,-5,-7,-63,-33,-35,-37,-41,-47,-51,-56,-60,-61,-62,99,100,103,-32,-57,-64,114,-34,-36,-38,-42,-48,-52,-65,118,]),'{':([14,19,20,23,24,25,27,29,31,33,34,35,36,37,38,42,44,53,70,79,99,100,112,115,117,118,119,],[19,-10,27,29,-10,-12,-10,29,29,-22,-23,-24,-25,-26,-27,-11,29,-13,-28,-32,29,29,-30,29,-29,29,-31,]),';':([19,23,24,25,27,29,31,33,34,35,36,37,38,42,43,44,48,53,55,56,57,58,59,60,61,62,63,65,66,67,70,72,79,97,99,100,101,103,105,106,107,108,109,110,112,114,115,117,118,119,],[-10,37,-10,-12,-10,37,37,-22,-23,-24,-25,-26,-27,-11,53,37,73,-13,-63,79,-33,-35,-37,-41,-47,-51,-56,-60,-61,-62,-28,98,-32,-57,37,37,113,-64,-34,-36,-38,-42,-48,-52,-30,-65,37,-29,37,-31,]),'IF':([19,23,24,25,27,29,31,33,34,35,36,37,38,42,44,53,70,79,99,100,112,115,117,118,119,],[-10,39,-10,-12,-10,39,39,-22,-23,-24,-25,-26,-27,-11,39,-13,-28,-32,39,39,-30,39,-29,39,-31,]),'WHILE':([19,23,24,25,27,29,31,33,34,35,36,37,38,42,44,53,70,79,99,100,112,115,117,118,119,],[-10,40,-10,-12,-10,40,40,-22,-23,-24,-25,-26,-27,-11,40,-13,-28,-32,40,40,-30,40,-29,40,-31,]),'FOR':([19,23,24,25,27,29,31,33,34,35,36,37,38,42,44,53,70,79,99,100,112,115,117,118,119,],[-10,41,-10,-12,-10,41,41,-22,-23,-24,-25,-26,-27,-11,41,-13,-28,-32,41,41,-30,41,-29,41,-31,]),'RETURN':([19,23,24,25,27,30,31,32,33,34,35,36,37,38,42,44,49,53,54,70,79,112,117,119,],[-10,-10,-10,-12,-10,48,-10,-21,-22,-23,-24,-25,-26,-27,-11,-10,-20,-13,48,-28,-32,-30,-29,-31,]),'=':([28,],[45,]),'}':([29,31,32,33,34,35,36,37,38,46,47,49,70,73,77,79,98,112,117,119,],[-10,-10,-21,-22,-23,-24,-25,-26,-27,70,71,-20,-28,-9,102,-32,-8,-30,-29,-31,]),'ELSE':([33,34,35,36,37,38,70,79,111,112,117,119,],[-22,-23,-24,-25,-26,-27,-28,-32,115,-30,-29,-31,]),'INTLIT':([45,48,50,51,64,68,69,76,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,],[65,65,65,65,65,-58,-59,65,65,-32,65,65,65,-39,-40,65,65,-43,-44,-45,-46,-49,-50,65,-53,-54,-55,]),'FLOATLIT':([45,48,50,51,64,68,69,76,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,],[66,66,66,66,66,-58,-59,66,66,-32,66,66,66,-39,-40,66,66,-43,-44,-45,-46,-49,-50,66,-53,-54,-55,]),'BOOLLIT':([45,48,50,51,64,68,69,76,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,],[67,67,67,67,67,-58,-59,67,67,-32,67,67,67,-39,-40,67,67,-43,-44,-45,-46,-49,-50,67,-53,-54,-55,]),'-':([45,48,50,51,55,60,61,62,63,65,66,67,76,79,80,81,82,83,84,85,86,87,88,89,90,91,92,97,103,108,109,110,114,],[68,68,68,68,-63,92,-47,-51,-56,-60,-61,-62,68,-32,68,68,68,-39,-40,68,68,-43,-44,-45,-46,-49,-50,-57,-64,92,-48,-52,-65,]),'!':([45,48,50,51,76,79,80,81,82,83,84,85,86,87,88,89,90,91,92,],[69,69,69,69,69,-32,69,69,69,-39,-40,69,69,-43,-44,-45,-46,-49,-50,]),'*':([55,61,62,63,65,66,67,97,103,109,110,114,],[-63,94,-51,-56,-60,-61,-62,-57,-64,94,-52,-65,]),'/':([55,61,62,63,65,66,67,97,103,109,110,114,],[-63,95,-51,-56,-60,-61,-62,-57,-64,95,-52,-65,]),'%':([55,61,62,63,65,66,67,97,103,109,110,114,],[-63,96,-51,-56,-60,-61,-62,-57,-64,96,-52,-65,]),'<':([55,60,61,62,63,65,66,67,97,103,109,110,114,],[-63,87,-47,-51,-56,-60,-61,-62,-57,-64,-48,-52,-65,]),'LTE':([55,60,61,62,63,65,66,67,97,103,109,110,114,],[-63,88,-47,-51,-56,-60,-61,-62,-57,-64,-48,-52,-65,]),'>':([55,60,61,62,63,65,66,67,97,103,109,110,114,],[-63,89,-47,-51,-56,-60,-61,-62,-57,-64,-48,-52,-65,]),'GTE':([55,60,61,62,63,65,66,67,97,103,109,110,114,],[-63,90,-47,-51,-56,-60,-61,-62,-57,-64,-48,-52,-65,]),'+':([55,60,61,62,63,65,66,67,97,103,108,109,110,114,],[-63,91,-47,-51,-56,-60,-61,-62,-57,-64,91,-48,-52,-65,]),'EQ':([55,59,60,61,62,63,65,66,67,97,103,108,109,110,114,],[-63,83,-41,-47,-51,-56,-60,-61,-62,-57,-64,-42,-48,-52,-65,]),'NEQ':([55,59,60,61,62,63,65,66,67,97,103,108,109,110,114,],[-63,84,-41,-47,-51,-56,-60,-61,-62,-57,-64,-42,-48,-52,-65,]),'AND':([55,57,58,59,60,61,62,63,65,66,67,97,103,105,106,107,108,109,110,114,],[-63,81,-35,-37,-41,-47,-51,-56,-60,-61,-62,-57,-64,81,-36,-38,-42,-48,-52,-65,]),'OR':([55,56,57,58,59,60,61,62,63,65,66,67,72,74,75,97,101,103,105,106,107,108,109,110,114,],[-63,80,-33,-35,-37,-41,-47,-51,-56,-60,-61,-62,80,80,80,-57,80,-64,-34,-36,-38,-42,-48,-52,-65,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'Program':([0,2,],[1,11,]),'Function':([0,2,],[2,2,]),'empty':([0,2,13,16,19,23,24,27,29,31,44,],[3,3,17,17,25,32,25,25,32,32,32,]),'FunctionReturnType':([0,2,],[4,4,]),'Type':([0,2,13,16,19,24,27,],[5,5,18,18,26,26,26,]),'Params':([13,16,],[15,21,]),'Param':([13,16,],[16,16,]),'Declarations':([19,24,27,],[23,42,44,]),'Declaration':([19,24,27,],[24,24,24,]),'Statements':([23,29,31,44,],[30,46,49,54,]),'Statement':([23,29,31,44,99,100,115,118,],[31,31,31,31,111,112,117,119,]),'Assignment':([23,29,31,44,52,99,100,113,115,118,],[33,33,33,33,76,33,33,116,33,33,]),'IfStatement':([23,29,31,44,99,100,115,118,],[34,34,34,34,34,34,34,34,]),'WhileStatement':([23,29,31,44,99,100,115,118,],[35,35,35,35,35,35,35,35,]),'ForStatement':([23,29,31,44,99,100,115,118,],[36,36,36,36,36,36,36,36,]),'Block':([23,29,31,44,99,100,115,118,],[38,38,38,38,38,38,38,38,]),'ReturnStatement':([30,54,],[47,77,]),'Expression':([45,48,50,51,76,],[56,72,74,75,101,]),'Conjunction':([45,48,50,51,76,80,],[57,57,57,57,57,105,]),'Equality':([45,48,50,51,76,80,81,],[58,58,58,58,58,58,106,]),'Relation':([45,48,50,51,76,80,81,82,],[59,59,59,59,59,59,59,107,]),'Addition':([45,48,50,51,76,80,81,82,85,],[60,60,60,60,60,60,60,60,108,]),'Term':([45,48,50,51,76,80,81,82,85,86,],[61,61,61,61,61,61,61,61,61,109,]),'Factor':([45,48,50,51,76,80,81,82,85,86,],[62,62,62,62,62,62,62,62,62,62,]),'Primary':([45,48,50,51,64,76,78,80,81,82,85,86,93,],[63,63,63,63,97,63,104,63,63,63,63,63,110,]),'UnaryOp':([45,48,50,51,76,80,81,82,85,86,],[64,64,64,64,64,64,64,64,64,64,]),'EquOp':([59,],[82,]),'RelOp':([60,],[85,]),'AddOp':([60,108,],[86,86,]),'MulOp':([61,109,],[93,93,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> Program","S'",1,None,None,None),
  ('Program -> Function Program','Program',2,'p_Program','clase0206.py',65),
  ('Program -> empty','Program',1,'p_Program','clase0206.py',66),
  ('Function -> FunctionReturnType ID ( ) { Declarations Statements ReturnStatement }','Function',9,'p_Function','clase0206.py',73),
  ('Function -> FunctionReturnType ID ( Params ) { Declarations Statements ReturnStatement }','Function',10,'p_Function','clase0206.py',74),
  ('Params -> Param Params','Params',2,'p_Params','clase0206.py',83),
  ('Params -> empty','Params',1,'p_Params','clase0206.py',84),
  ('Param -> Type ID','Param',2,'p_Param','clase0206.py',92),
  ('ReturnStatement -> RETURN Expression ;','ReturnStatement',3,'p_ReturnStatement','clase0206.py',99),
  ('ReturnStatement -> RETURN ;','ReturnStatement',2,'p_ReturnStatement','clase0206.py',100),
  ('empty -> <empty>','empty',0,'p_empty','clase0206.py',109),
  ('Declarations -> Declaration Declarations','Declarations',2,'p_Declarations','clase0206.py',115),
  ('Declarations -> empty','Declarations',1,'p_Declarations','clase0206.py',116),
  ('Declaration -> Type ID ;','Declaration',3,'p_Declaration','clase0206.py',123),
  ('Type -> INT','Type',1,'p_Type','clase0206.py',129),
  ('Type -> BOOL','Type',1,'p_Type','clase0206.py',130),
  ('Type -> FLOAT','Type',1,'p_Type','clase0206.py',131),
  ('Type -> CHAR','Type',1,'p_Type','clase0206.py',132),
  ('FunctionReturnType -> Type','FunctionReturnType',1,'p_FunctionReturnType','clase0206.py',138),
  ('FunctionReturnType -> VOID','FunctionReturnType',1,'p_FunctionReturnType','clase0206.py',139),
  ('Statements -> Statement Statements','Statements',2,'p_Statements','clase0206.py',145),
  ('Statements -> empty','Statements',1,'p_Statements','clase0206.py',146),
  ('Statement -> Assignment','Statement',1,'p_Statement','clase0206.py',153),
  ('Statement -> IfStatement','Statement',1,'p_Statement','clase0206.py',154),
  ('Statement -> WhileStatement','Statement',1,'p_Statement','clase0206.py',155),
  ('Statement -> ForStatement','Statement',1,'p_Statement','clase0206.py',156),
  ('Statement -> ;','Statement',1,'p_Statement','clase0206.py',157),
  ('Statement -> Block','Statement',1,'p_Statement','clase0206.py',158),
  ('Block -> { Statements }','Block',3,'p_Block','clase0206.py',164),
  ('IfStatement -> IF ( Expression ) Statement ELSE Statement','IfStatement',7,'p_IfStatement','clase0206.py',170),
  ('WhileStatement -> WHILE ( Expression ) Statement','WhileStatement',5,'p_WhileStatement','clase0206.py',176),
  ('ForStatement -> FOR ( Assignment Expression ; Assignment ) Statement','ForStatement',8,'p_ForStatement','clase0206.py',182),
  ('Assignment -> ID = Expression ;','Assignment',4,'p_Assignment','clase0206.py',188),
  ('Expression -> Conjunction','Expression',1,'p_Expression','clase0206.py',194),
  ('Expression -> Expression OR Conjunction','Expression',3,'p_Expression','clase0206.py',195),
  ('Conjunction -> Equality','Conjunction',1,'p_Conjunction','clase0206.py',204),
  ('Conjunction -> Conjunction AND Equality','Conjunction',3,'p_Conjunction','clase0206.py',205),
  ('Equality -> Relation','Equality',1,'p_Equality','clase0206.py',214),
  ('Equality -> Relation EquOp Relation','Equality',3,'p_Equality','clase0206.py',215),
  ('EquOp -> EQ','EquOp',1,'p_EquOp','clase0206.py',224),
  ('EquOp -> NEQ','EquOp',1,'p_EquOp','clase0206.py',225),
  ('Relation -> Addition','Relation',1,'p_Relation','clase0206.py',231),
  ('Relation -> Addition RelOp Addition','Relation',3,'p_Relation','clase0206.py',232),
  ('RelOp -> <','RelOp',1,'p_RelOp','clase0206.py',241),
  ('RelOp -> LTE','RelOp',1,'p_RelOp','clase0206.py',242),
  ('RelOp -> >','RelOp',1,'p_RelOp','clase0206.py',243),
  ('RelOp -> GTE','RelOp',1,'p_RelOp','clase0206.py',244),
  ('Addition -> Term','Addition',1,'p_Addition','clase0206.py',250),
  ('Addition -> Addition AddOp Term','Addition',3,'p_Addition','clase0206.py',251),
  ('AddOp -> +','AddOp',1,'p_AddOp','clase0206.py',260),
  ('AddOp -> -','AddOp',1,'p_AddOp','clase0206.py',261),
  ('Term -> Factor','Term',1,'p_Term','clase0206.py',267),
  ('Term -> Term MulOp Primary','Term',3,'p_Term','clase0206.py',268),
  ('MulOp -> *','MulOp',1,'p_MulOp','clase0206.py',277),
  ('MulOp -> /','MulOp',1,'p_MulOp','clase0206.py',278),
  ('MulOp -> %','MulOp',1,'p_MulOp','clase0206.py',279),
  ('Factor -> Primary','Factor',1,'p_Factor','clase0206.py',285),
  ('Factor -> UnaryOp Primary','Factor',2,'p_Factor','clase0206.py',286),
  ('UnaryOp -> -','UnaryOp',1,'p_UnaryOp','clase0206.py',296),
  ('UnaryOp -> !','UnaryOp',1,'p_UnaryOp','clase0206.py',297),
  ('Primary -> INTLIT','Primary',1,'p_Primary_IntLit','clase0206.py',302),
  ('Primary -> FLOATLIT','Primary',1,'p_Primary_FloatLit','clase0206.py',306),
  ('Primary -> BOOLLIT','Primary',1,'p_Primary_BoolLit','clase0206.py',310),
  ('Primary -> ID','Primary',1,'p_Primary_Id','clase0206.py',314),
  ('Primary -> ID ( )','Primary',3,'p_Primary_FunctionCall','clase0206.py',320),
  ('Primary -> ID ( Primary )','Primary',4,'p_Primary_FunctionCall','clase0206.py',321),
]
