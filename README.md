# clite-compiler

**A compiler based on clite grammar**

For the class of Desarrollo de Aplicaciones Avanzadas de Ciencias Computacionales I made a compiler based on the 
[clite grammar](https://myslu.stlawu.edu/~ehar/Spring10/364/clite_grammar.html) (a simple grammar based on C and Java)
plus additional functionalities that were not present on the original grammar. 
It is a perfect project to know more about the process of compilation and about how all programming languages work.

## How it works:
The compiler uses ply.lex for lexical analysis and ply.yacc for syntactical analysis. It creates llvm code with the help of
a visitor design pattern and the library llvmlite

## Features
- You can do assignations
- You can do mathematical operations (addition, substraction, multiplication, division)
- You can use floats
- You can define control structures (ifs, fors, while)
- You can define functions
- You have type casting
- You can make operations with the results of functions
- You have negations

## Libraries used
- Ply.lex
- Ply.yacc
- llvmlite
- runtime
