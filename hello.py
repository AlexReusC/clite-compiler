# %%
from llvmlite import ir

intType = ir.IntType(64)
module = ir.Module(name="prog")

# int main() {
fnty = ir.FunctionType(intType, [])
func = ir.Function(module, fnty, name='main')

entry = func.append_basic_block('entry')
builder = ir.IRBuilder(entry)

# return 5 + 3; }
lhs = ir.Constant(intType, 5)
rhs = intType(3)
temp = builder.add(lhs, rhs)
builder.ret(temp)

print(module)

# %%
import runtime as rt
from ctypes import CFUNCTYPE, c_int

engine = rt.create_execution_engine()
mod = rt.compile_ir(engine, str(module))
func_ptr = engine.get_function_address("main")

# Run the function via ctypes
cfunc = CFUNCTYPE(c_int)(func_ptr)
res = cfunc()
print("main() =", res)
print(mod)

# %%
from llvmlite import binding as llvm
llvm.initialize()
llvm.initialize_native_asmprinter()
llvm.initialize_native_target()
target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine(codemodel='default')
mod = llvm.parse_assembly(str(module))
object = open("test.o", "wb")   
object.write(target_machine.emit_object(mod))
object.close()


# %%
