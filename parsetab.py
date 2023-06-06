
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = "AND BOOL CHAR ELSE EQ FLOAT FOR GTE ID IF INT INTLIT LTE NEQ OR RETURN WHILE\n    Program : Function Program\n            | empty\n    \n    Function : Type ID '(' ')' '{' Declarations Statements ReturnStatement '}'\n    \n    ReturnStatement : RETURN Expression ';'\n    \n    empty :\n    \n    Declarations : Declaration Declarations\n                 | empty\n    \n    Declaration : Type ID ';'\n    \n    Type : INT\n         | BOOL\n         | FLOAT\n         | CHAR\n    \n    Statements : Statement Statements\n               | empty\n    \n    Statement : Assignment\n              | IfStatement\n              | WhileStatement\n              | ForStatement\n              | ';'\n              | Block\n    \n    Block : '{' Statements '}'\n    \n    IfStatement : IF '(' Expression ')' Statement ELSE Statement\n    \n    WhileStatement : WHILE '(' Expression ')' Statement\n    \n    ForStatement : FOR '(' Assignment Expression ';' Assignment ')' Statement\n    \n    Assignment : ID '=' Expression ';'\n    \n    Expression : Conjunction\n               | Expression OR Conjunction\n    \n    Conjunction : Equality\n                | Conjunction AND Equality\n    \n    Equality : Relation\n             | Relation EquOp Relation\n    \n    EquOp : EQ\n          | NEQ\n    \n    Relation : Addition\n             | Addition RelOp Addition \n    \n    RelOp : '<'\n          | LTE\n          | '>'\n          | GTE\n    \n    Addition : Term\n             | Addition AddOp Term\n    \n    AddOp : '+'\n          | '-'\n    \n    Term : Factor\n         | Term MulOp Primary\n    \n    MulOp : '*'\n          | '/'\n          | '%'\n    \n    Factor : Primary\n           | UnaryOp Primary\n    \n    UnaryOp : '-'\n            | '!'\n    Primary : INTLITPrimary : ID\n    Primary : ID '(' ')'\n    "
    
_lr_action_items = {'$end':([0,1,2,3,9,57,],[-5,0,-5,-2,-1,-3,]),'INT':([0,2,13,16,34,57,],[5,5,5,5,-8,-3,]),'BOOL':([0,2,13,16,34,57,],[6,6,6,6,-8,-3,]),'FLOAT':([0,2,13,16,34,57,],[7,7,7,7,-8,-3,]),'CHAR':([0,2,13,16,34,57,],[8,8,8,8,-8,-3,]),'ID':([4,5,6,7,8,13,14,15,16,17,20,22,24,25,26,27,28,29,33,34,35,38,40,41,42,52,54,55,56,61,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,83,84,94,95,96,98,99,100,],[10,-9,-10,-11,-12,-5,18,19,-5,-7,19,19,-15,-16,-17,-18,-19,-20,-6,-8,43,43,43,43,19,43,-51,-52,-21,43,-25,43,43,43,-32,-33,43,43,-36,-37,-38,-39,-42,-43,43,-46,-47,-48,19,19,-23,19,19,-22,19,-24,]),'(':([10,30,31,32,43,],[11,40,41,42,62,]),')':([11,43,45,46,47,48,49,50,51,53,59,60,62,63,81,86,87,88,89,90,91,92,97,],[12,-54,-26,-28,-30,-34,-40,-44,-49,-53,83,84,86,-25,-50,-55,-27,-29,-31,-35,-41,-45,99,]),'{':([12,13,15,16,17,20,22,24,25,26,27,28,29,33,34,56,63,83,84,94,96,98,99,100,],[13,-5,20,-5,-7,20,20,-15,-16,-17,-18,-19,-20,-6,-8,-21,-25,20,20,-23,20,-22,20,-24,]),';':([13,15,16,17,18,20,22,24,25,26,27,28,29,33,34,43,44,45,46,47,48,49,50,51,53,56,58,63,81,83,84,85,86,87,88,89,90,91,92,94,96,98,99,100,],[-5,28,-5,-7,34,28,28,-15,-16,-17,-18,-19,-20,-6,-8,-54,63,-26,-28,-30,-34,-40,-44,-49,-53,-21,82,-25,-50,28,28,95,-55,-27,-29,-31,-35,-41,-45,-23,28,-22,28,-24,]),'IF':([13,15,16,17,20,22,24,25,26,27,28,29,33,34,56,63,83,84,94,96,98,99,100,],[-5,30,-5,-7,30,30,-15,-16,-17,-18,-19,-20,-6,-8,-21,-25,30,30,-23,30,-22,30,-24,]),'WHILE':([13,15,16,17,20,22,24,25,26,27,28,29,33,34,56,63,83,84,94,96,98,99,100,],[-5,31,-5,-7,31,31,-15,-16,-17,-18,-19,-20,-6,-8,-21,-25,31,31,-23,31,-22,31,-24,]),'FOR':([13,15,16,17,20,22,24,25,26,27,28,29,33,34,56,63,83,84,94,96,98,99,100,],[-5,32,-5,-7,32,32,-15,-16,-17,-18,-19,-20,-6,-8,-21,-25,32,32,-23,32,-22,32,-24,]),'RETURN':([13,15,16,17,21,22,23,24,25,26,27,28,29,33,34,39,56,63,94,98,100,],[-5,-5,-5,-7,38,-5,-14,-15,-16,-17,-18,-19,-20,-6,-8,-13,-21,-25,-23,-22,-24,]),'=':([19,],[35,]),'}':([20,22,23,24,25,26,27,28,29,36,37,39,56,63,82,94,98,100,],[-5,-5,-14,-15,-16,-17,-18,-19,-20,56,57,-13,-21,-25,-4,-23,-22,-24,]),'ELSE':([24,25,26,27,28,29,56,63,93,94,98,100,],[-15,-16,-17,-18,-19,-20,-21,-25,96,-23,-22,-24,]),'INTLIT':([35,38,40,41,52,54,55,61,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,],[53,53,53,53,53,-51,-52,53,-25,53,53,53,-32,-33,53,53,-36,-37,-38,-39,-42,-43,53,-46,-47,-48,]),'-':([35,38,40,41,43,48,49,50,51,53,61,63,64,65,66,67,68,69,70,71,72,73,74,75,76,81,86,90,91,92,],[54,54,54,54,-54,76,-40,-44,-49,-53,54,-25,54,54,54,-32,-33,54,54,-36,-37,-38,-39,-42,-43,-50,-55,76,-41,-45,]),'!':([35,38,40,41,61,63,64,65,66,67,68,69,70,71,72,73,74,75,76,],[55,55,55,55,55,-25,55,55,55,-32,-33,55,55,-36,-37,-38,-39,-42,-43,]),'*':([43,49,50,51,53,81,86,91,92,],[-54,78,-44,-49,-53,-50,-55,78,-45,]),'/':([43,49,50,51,53,81,86,91,92,],[-54,79,-44,-49,-53,-50,-55,79,-45,]),'%':([43,49,50,51,53,81,86,91,92,],[-54,80,-44,-49,-53,-50,-55,80,-45,]),'<':([43,48,49,50,51,53,81,86,91,92,],[-54,71,-40,-44,-49,-53,-50,-55,-41,-45,]),'LTE':([43,48,49,50,51,53,81,86,91,92,],[-54,72,-40,-44,-49,-53,-50,-55,-41,-45,]),'>':([43,48,49,50,51,53,81,86,91,92,],[-54,73,-40,-44,-49,-53,-50,-55,-41,-45,]),'GTE':([43,48,49,50,51,53,81,86,91,92,],[-54,74,-40,-44,-49,-53,-50,-55,-41,-45,]),'+':([43,48,49,50,51,53,81,86,90,91,92,],[-54,75,-40,-44,-49,-53,-50,-55,75,-41,-45,]),'EQ':([43,47,48,49,50,51,53,81,86,90,91,92,],[-54,67,-34,-40,-44,-49,-53,-50,-55,-35,-41,-45,]),'NEQ':([43,47,48,49,50,51,53,81,86,90,91,92,],[-54,68,-34,-40,-44,-49,-53,-50,-55,-35,-41,-45,]),'AND':([43,45,46,47,48,49,50,51,53,81,86,87,88,89,90,91,92,],[-54,65,-28,-30,-34,-40,-44,-49,-53,-50,-55,65,-29,-31,-35,-41,-45,]),'OR':([43,44,45,46,47,48,49,50,51,53,58,59,60,81,85,86,87,88,89,90,91,92,],[-54,64,-26,-28,-30,-34,-40,-44,-49,-53,64,64,64,-50,64,-55,-27,-29,-31,-35,-41,-45,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'Program':([0,2,],[1,9,]),'Function':([0,2,],[2,2,]),'empty':([0,2,13,15,16,20,22,],[3,3,17,23,17,23,23,]),'Type':([0,2,13,16,],[4,4,14,14,]),'Declarations':([13,16,],[15,33,]),'Declaration':([13,16,],[16,16,]),'Statements':([15,20,22,],[21,36,39,]),'Statement':([15,20,22,83,84,96,99,],[22,22,22,93,94,98,100,]),'Assignment':([15,20,22,42,83,84,95,96,99,],[24,24,24,61,24,24,97,24,24,]),'IfStatement':([15,20,22,83,84,96,99,],[25,25,25,25,25,25,25,]),'WhileStatement':([15,20,22,83,84,96,99,],[26,26,26,26,26,26,26,]),'ForStatement':([15,20,22,83,84,96,99,],[27,27,27,27,27,27,27,]),'Block':([15,20,22,83,84,96,99,],[29,29,29,29,29,29,29,]),'ReturnStatement':([21,],[37,]),'Expression':([35,38,40,41,61,],[44,58,59,60,85,]),'Conjunction':([35,38,40,41,61,64,],[45,45,45,45,45,87,]),'Equality':([35,38,40,41,61,64,65,],[46,46,46,46,46,46,88,]),'Relation':([35,38,40,41,61,64,65,66,],[47,47,47,47,47,47,47,89,]),'Addition':([35,38,40,41,61,64,65,66,69,],[48,48,48,48,48,48,48,48,90,]),'Term':([35,38,40,41,61,64,65,66,69,70,],[49,49,49,49,49,49,49,49,49,91,]),'Factor':([35,38,40,41,61,64,65,66,69,70,],[50,50,50,50,50,50,50,50,50,50,]),'Primary':([35,38,40,41,52,61,64,65,66,69,70,77,],[51,51,51,51,81,51,51,51,51,51,51,92,]),'UnaryOp':([35,38,40,41,61,64,65,66,69,70,],[52,52,52,52,52,52,52,52,52,52,]),'EquOp':([47,],[66,]),'RelOp':([48,],[69,]),'AddOp':([48,90,],[70,70,]),'MulOp':([49,91,],[77,77,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> Program","S'",1,None,None,None),
  ('Program -> Function Program','Program',2,'p_Program','clase0206.py',55),
  ('Program -> empty','Program',1,'p_Program','clase0206.py',56),
  ('Function -> Type ID ( ) { Declarations Statements ReturnStatement }','Function',9,'p_Function','clase0206.py',63),
  ('ReturnStatement -> RETURN Expression ;','ReturnStatement',3,'p_ReturnStatement','clase0206.py',69),
  ('empty -> <empty>','empty',0,'p_empty','clase0206.py',76),
  ('Declarations -> Declaration Declarations','Declarations',2,'p_Declarations','clase0206.py',82),
  ('Declarations -> empty','Declarations',1,'p_Declarations','clase0206.py',83),
  ('Declaration -> Type ID ;','Declaration',3,'p_Declaration','clase0206.py',90),
  ('Type -> INT','Type',1,'p_Type','clase0206.py',96),
  ('Type -> BOOL','Type',1,'p_Type','clase0206.py',97),
  ('Type -> FLOAT','Type',1,'p_Type','clase0206.py',98),
  ('Type -> CHAR','Type',1,'p_Type','clase0206.py',99),
  ('Statements -> Statement Statements','Statements',2,'p_Statements','clase0206.py',105),
  ('Statements -> empty','Statements',1,'p_Statements','clase0206.py',106),
  ('Statement -> Assignment','Statement',1,'p_Statement','clase0206.py',113),
  ('Statement -> IfStatement','Statement',1,'p_Statement','clase0206.py',114),
  ('Statement -> WhileStatement','Statement',1,'p_Statement','clase0206.py',115),
  ('Statement -> ForStatement','Statement',1,'p_Statement','clase0206.py',116),
  ('Statement -> ;','Statement',1,'p_Statement','clase0206.py',117),
  ('Statement -> Block','Statement',1,'p_Statement','clase0206.py',118),
  ('Block -> { Statements }','Block',3,'p_Block','clase0206.py',124),
  ('IfStatement -> IF ( Expression ) Statement ELSE Statement','IfStatement',7,'p_IfStatement','clase0206.py',130),
  ('WhileStatement -> WHILE ( Expression ) Statement','WhileStatement',5,'p_WhileStatement','clase0206.py',136),
  ('ForStatement -> FOR ( Assignment Expression ; Assignment ) Statement','ForStatement',8,'p_ForStatement','clase0206.py',142),
  ('Assignment -> ID = Expression ;','Assignment',4,'p_Assignment','clase0206.py',148),
  ('Expression -> Conjunction','Expression',1,'p_Expression','clase0206.py',154),
  ('Expression -> Expression OR Conjunction','Expression',3,'p_Expression','clase0206.py',155),
  ('Conjunction -> Equality','Conjunction',1,'p_Conjunction','clase0206.py',164),
  ('Conjunction -> Conjunction AND Equality','Conjunction',3,'p_Conjunction','clase0206.py',165),
  ('Equality -> Relation','Equality',1,'p_Equality','clase0206.py',174),
  ('Equality -> Relation EquOp Relation','Equality',3,'p_Equality','clase0206.py',175),
  ('EquOp -> EQ','EquOp',1,'p_EquOp','clase0206.py',184),
  ('EquOp -> NEQ','EquOp',1,'p_EquOp','clase0206.py',185),
  ('Relation -> Addition','Relation',1,'p_Relation','clase0206.py',191),
  ('Relation -> Addition RelOp Addition','Relation',3,'p_Relation','clase0206.py',192),
  ('RelOp -> <','RelOp',1,'p_RelOp','clase0206.py',201),
  ('RelOp -> LTE','RelOp',1,'p_RelOp','clase0206.py',202),
  ('RelOp -> >','RelOp',1,'p_RelOp','clase0206.py',203),
  ('RelOp -> GTE','RelOp',1,'p_RelOp','clase0206.py',204),
  ('Addition -> Term','Addition',1,'p_Addition','clase0206.py',210),
  ('Addition -> Addition AddOp Term','Addition',3,'p_Addition','clase0206.py',211),
  ('AddOp -> +','AddOp',1,'p_AddOp','clase0206.py',220),
  ('AddOp -> -','AddOp',1,'p_AddOp','clase0206.py',221),
  ('Term -> Factor','Term',1,'p_Term','clase0206.py',227),
  ('Term -> Term MulOp Primary','Term',3,'p_Term','clase0206.py',228),
  ('MulOp -> *','MulOp',1,'p_MulOp','clase0206.py',237),
  ('MulOp -> /','MulOp',1,'p_MulOp','clase0206.py',238),
  ('MulOp -> %','MulOp',1,'p_MulOp','clase0206.py',239),
  ('Factor -> Primary','Factor',1,'p_Factor','clase0206.py',245),
  ('Factor -> UnaryOp Primary','Factor',2,'p_Factor','clase0206.py',246),
  ('UnaryOp -> -','UnaryOp',1,'p_UnaryOp','clase0206.py',262),
  ('UnaryOp -> !','UnaryOp',1,'p_UnaryOp','clase0206.py',263),
  ('Primary -> INTLIT','Primary',1,'p_Primary_IntLit','clase0206.py',268),
  ('Primary -> ID','Primary',1,'p_Primary_Id','clase0206.py',272),
  ('Primary -> ID ( )','Primary',3,'p_Primary_FunctionCall','clase0206.py',277),
]
