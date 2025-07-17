#!/usr/bin/env python3
import os
import sys
import subprocess

# Order of redops, tys, protos, algos must match src/include/device.h
all_colls =  ["AllGather","AllReduce","AllToAllPivot","Broadcast","Reduce","ReduceScatter","SendRecv"]
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys =    ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16","f8e4m3","f8e5m2"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos =  ["TREE","RING", "PAT"]
all_unroll = ["1", "2", "4"]

all_params = [all_colls, all_algos, all_protos, all_redops, all_tys, all_unroll]

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
    #os.truncate(os.path.join(gensrc, name), 0)
else:
  os.makedirs(gensrc)

################################################################################
# The command line argument is used as a regex to filter the functions
# which make it into librccl. This is helpful for reducing the binary when
# developing device code. The regex supports non-space containing globs '*',
# and union 'a|b'. The string representing the function has the form:
#
# <coll> <algo> <proto> <redop> <type> <unroll>
#
# The possible values for redop, type, algo, proto can be found in the all_<foo>
# lists at the top of this file.
#
# Example use-cases:
#
# # Only send/recv:
# make ONLY_FUNCS="SendRecv"
#
# # Only AllReduce and Reduce
# make ONLY_FUNCS="AllReduce|Reduce"
#
# # Only AllGather with unroll=4
# make ONLY_FUNCS="AllGather * * * * 4"
#
# # Only non-reductions:
# make ONLY_FUNCS="AllGather * *|Broadcast * *|SendRecv"
#
# # Only AllReduce Sum int32_t (but all algos, protos)
# make ONLY_FUNCS="AllReduce * * Sum i32"
#
# # Only AllReduce RING Max float (but all protos and unrolls)
# make ONLY_FUNCS="AllReduce RING * Max f32"
#
# # AllReduce TREE LL128 Prod rccl_bfloat16 unroll=1
# make ONLY_FUNCS="AllReduce TREE LL128 Prod bf16 1"
#
# # AllReduce RING SIMPLE and ReduceScatter RING LL float (but all redops, types, unrolls for AllReduce and all redops, unrolls for ReduceScatter)
# make ONLY_FUNCS="AllReduce RING SIMPLE * *|ReduceScatter RING LL * f32 *"
#                         --- or ---
# make ONLY_FUNCS="AllReduce RING SIMPLE|ReduceScatter RING LL * f32 *"
#
#
# make ONLY_FUNCS="AllReduce RING/TREE LL/LL128/SIMPLE Sum/MinMax i8/u8/f16/f32/f64/bf16/f8e4m3/f8e5m2 1/2/4|AllGather RING LL/LL128/SIMPLE Sum i8 1/2/4|AllToAllPivot RING SIMPLE Sum i8 1/2/4|Broadcast RING LL/LL128/SIMPLE Sum i8 1/2/4|Reduce RING LL/LL128/SIMPLE Sum/MinMax i8/u8/f16/f32/f64/bf16/f8e4m3/f8e5m2 1/2/4|ReduceScatter RING LL/LL128/SIMPLE Sum/MinMax i8/u8/f16/f32/f64/bf16/f8e4m3/f8e5m2 1/2/4|SendRecv RING SIMPLE Sum i8 1/2/4"
#
# # ONLY_FUNCS can be used together for debugging

# Paste all non-None arguments together with `sep`.
def paste(sep, *args):
  return sep.join(x for x in args if x is not None)

is_ifc             = 1 if sys.argv[2] == "ON" else 0
is_colltrace       = 1 if sys.argv[3] == "ON" else 0
is_msccl_kernels   = 1 if sys.argv[4] == "ON" else 0

func_pattern = sys.argv[5:]
if func_pattern and func_pattern[0]:
  func_pattern = func_pattern[0]
else:
  func_pattern = "AllGather|AllReduce|AllToAllPivot|Broadcast|Reduce|ReduceScatter|SendRecv"

################################################################################

algos_of_coll = {
  "AllGather":     ["RING", "PAT"],
  "AllReduce":     ["RING", "TREE"],
  "AllToAllPivot": ["RING"],
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING", "PAT"],
  "SendRecv":      ["RING"]
}

protos_of_coll = {
  "AllGather":     all_protos,
  "AllReduce":     all_protos,
  "AllToAllPivot": ["SIMPLE"],
  "Broadcast":     all_protos,
  "Reduce":        all_protos,
  "ReduceScatter": all_protos,
  "SendRecv":      ["SIMPLE"]
}

redops_of_coll = {
  "AllGather":     ["Sum"],
  "AllReduce":     all_redops,
  "AllToAllPivot": ["Sum"],
  "Broadcast":     ["Sum"],
  "Reduce":        all_redops,
  "ReduceScatter": all_redops,
  "SendRecv":      ["Sum"]
}

tys_of_coll = {
  "AllGather":     ["i8"],
  "AllReduce":     all_tys,
  "AllToAllPivot": ["i8"],
  "Broadcast":     ["i8"],
  "Reduce":        all_tys,
  "ReduceScatter": all_tys,
  "SendRecv":      ["i8"]
}

coll_camel_to_lower = {
  "AllGather":     "all_gather",
  "AllReduce":     "all_reduce",
  "AllToAllPivot": "alltoall_pivot",
  "Broadcast":     "broadcast",
  "Reduce":        "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv":      "sendrecv"
}
coll_lower_to_camel = {coll_camel_to_lower[x]: x for x in coll_camel_to_lower}

################################################################################

seen_unroll = []

# Helper function to check if the conditions for the collective is being met
def func_validate(coll, algo, proto, redop, ty, unroll):
  if redop == "SumPostDiv" and ty[0] not in ("i","u"):
    return False
  if algo not in algos_of_coll[coll] or proto not in protos_of_coll[coll] or redop not in redops_of_coll[coll] or ty not in tys_of_coll[coll] or unroll not in all_unroll:
    return False
  return True

# A recursive helper to generate collective functions based on the input given
def func_filter(function_params, current_idx, item_list=None):
  if item_list is None:
    item_list = []

  # Check if current_idx exceeds the max depth
  if current_idx < len(all_params):
    # Current element is the config parameter
    current_element = function_params[current_idx]

    # If the paramter is equal to '*', include all possible cases for it
    if current_element == "*":
      # all_params list must be in the same order as function_params --> <coll> <algo> <proto> <redop> <type> <unroll>
      # Get the current list from all_params
      current_list = all_params[current_idx]

      # Iterate over the items int the current_list
      for item in current_list:
        # Add item to item_list which will be used in the inner most loop
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, item_list)

        # For each loop layer remove the last element in item_list
        item_list.pop()
    else:
      # Check if the current element is recognized
      elements = current_element.split("/")
      current_param = all_params[current_idx]

      # Iterate over the elements in the elements list
      for item in elements:
        if item not in current_param:
          raise ValueError(f"Error: {item} is unrecognized or does not belong to this category {current_param}.")

      for item in elements:
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, item_list)

        # For each loop layer remove the last element in item_list
        item_list.pop()
  else:
    coll, algo, proto, redop, ty, unroll = item_list

    if func_validate(coll, algo, proto, redop, ty, unroll):
      if not unroll in seen_unroll:
        seen_unroll.append(unroll)
      yield(coll, algo, proto, redop, ty, unroll)

# Parse ONLY_FUNCS input and feed it to func_filter
def parse_input(func_pattern):
  input_list = sorted(func_pattern.split("|"))

  for input in input_list:
    function_params = input.split()
    params_length = len(function_params)

    # If a parameter is missing, append '*'
    while params_length < len(all_params):
      function_params.append("*")
      params_length += 1

    # Filter functions/kernels based on input
    yield from func_filter(function_params, 0)

# Maps functions to the chosen representative for the equivalence class it
# belongs to. For instance (sum, signed int) maps to (sum, unsigned int).
def equivalent_primary(coll, algo, proto, redop, ty, unroll):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    # map signed integer sum/prod to unsigned
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0]=="i":
      ty = "u"+ty[1:]
    # map signed integer min/max to unsigned for non-NVLS
    elif redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      ty = "u"+ty[1:]
  return (coll, algo, proto, redop, ty, unroll)

# Order rows are enumerated must match formula of `ncclDevFuncId()`:
def enumerate_func_rows():
  for unroll in all_unroll:
    for coll in all_colls:
      for algo in all_algos:
        for proto in all_protos:
          for redop in all_redops:
            for ty in all_tys:
              if func_validate(coll, algo, proto, redop, ty, unroll):
                yield (coll, algo, proto, redop, ty, unroll)

# Sort the hashmap based on custom key <coll> <algo> <proto> <redop> <ty> <unroll>
def custom_sort_key(fn):
    coll, algo, proto, redop, ty, unroll = fn

    return (
        all_unroll.index(unroll),
        all_colls.index(coll),
        all_algos.index(algo),
        all_protos.index(proto),
        all_redops.index(redop),
        all_tys.index(ty)
    )

################################################################################

# Corresponds to ncclDevFuncRowToId[]
func_rows = [fn for fn in enumerate_func_rows()]

# Corresponds to ncclDevFuncTable[]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in parse_input(func_pattern)), key=custom_sort_key)

# primary_to_index[primary_funcs[i]] == i
primary_to_index = {fn: primary_funcs.index(fn) if fn in primary_funcs else -1 for fn in func_rows}

################################################################################

# Generate <gensrc>/device_table.h
with open(os.path.join(gensrc, "device_table.h"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "device_table.h"))
  out = f.write

  out("#include \"common.h\"\n\n")

  if is_ifc: func_declaration = "__device__ void"
  else: func_declaration = "__device__ __attribute__((noinline)) void"

  for fn in primary_funcs:
    sym = paste("_", "ncclDevFunc", *fn)
    if fn[2] == "LL128":
      out("#if (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)) && defined(ENABLE_LL128)\n")
      out("%s %s();\n#else\n" % (func_declaration, sym))
      fn_ll = fn[:2] + ("LL",) + fn[3:]
      sym_ll = paste("_", "ncclDevFunc", *fn_ll)
      out("%s %s();\n#endif\n" % (func_declaration, sym_ll))
    else:
      out("%s %s();\n" % (func_declaration, sym))
  out("\n")

  out("typedef void(*ncclDevFuncPtr_t)();\n\n")

  # Generate function tables per unroll factor
  tableIdx = 0
  for curr_unroll in seen_unroll:
    out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable_%s[] = {\n" % curr_unroll)
    tableIdx = 0
    for fn in primary_funcs:
      coll, algo, proto, redop, ty, unroll = fn
      if curr_unroll != unroll: continue
      sym = paste("_", "ncclDevFunc", *fn)
      if fn[2] == "LL128":
        out("#if (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)) && defined(ENABLE_LL128)\n")
        out("/*%4d*/ %s,\n#else\n" % (tableIdx, sym))
        fn_ll = fn[:2] + ("LL",) + fn[3:]
        sym_ll = paste("_", "ncclDevFunc", *fn_ll)
        out("/*%4d*/ %s,\n#endif\n" % (tableIdx, sym_ll))
      else:
        out("/*%4d*/ %s,\n" % (tableIdx, sym))
      tableIdx += 1
    out("nullptr};\n")
    out("\n")

  # Construct indirection function workaround
  if not is_ifc:
    out("template<int unroll, unsigned short f, unsigned short l>\n"
      "struct Caller {\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call(unsigned short funcIndex) noexcept\n"
      "  {\n"
      "    constexpr unsigned short m = f + (l - f) / 2;\n"
      "    return (funcIndex < m) ? Caller<unroll, f, m>::call(funcIndex) : Caller<unroll, m, l>::call(funcIndex);\n"
      "  }\n"
      "};\n"
      "\n")

    for curr_unroll in seen_unroll:
      out("template<unsigned short f>\n")
      out("struct Caller<%s, f, f + 1>{\n" % curr_unroll)
      out("  static __forceinline__ __device__ __host__\n");
      out("  void call(unsigned short funcIndex) noexcept { ncclDevFuncTable_%s[f](); }\n" % curr_unroll)
      out("};\n")

  out("\n")
  # Create NCCL_CALL_FUNCTION helper function that will call the appropriate device function
  out("template <int unroll>\n"
      "__forceinline__ __device__ void NCCL_CALL_FUNCTIONS(unsigned short funcIndex) noexcept {\n")
  if is_ifc:
    for curr_unroll in seen_unroll:
      out("  if (unroll == %s) { ncclDevFuncTable_%s[funcIndex](); }\n" % (curr_unroll, curr_unroll))
  else:
    out(f"  Caller<unroll, 0, {tableIdx}>::call(funcIndex);\n")
  out("}\n\n")

  # Create RCCL
  out("template<int SpecializedFnId, typename SpecializedRunWorkBatch, bool COLLTRACE, int COLL_UNROLL>\n");
  out("__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args);\n\n");

  out("struct RunWorkNop {\n");
  out("  __device__ void run() {}\n");
  out("};\n\n");

  out("template <int UNROLL, bool COLLTRACE>\n"
      "__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void rcclGenericKernel(ncclDevKernelArgs4K const args4K) {\n"
      "  ncclKernelMain<-1, RunWorkNop, COLLTRACE, UNROLL>(&args4K.args);\n"
      "}\n\n")

  out("struct rcclKernelItem {\n");
  out("  void* funcPtr;\n");
  out("  int   unroll;\n");
  out("};\n\n");

  out("/* This table contains all the __global__ functions that were compiled */\n");
  out("static struct rcclKernelItem rcclKernelTable[] = {\n")
  for unroll in seen_unroll:
    out("  {(void*)&(rcclGenericKernel<%s, false>), %s},\n" % (unroll, unroll))
  out("#ifdef ENABLE_COLLTRACE\n")
  for unroll in seen_unroll:
    out("  {(void*)&(rcclGenericKernel<%s, true>), %s},\n" % (unroll, unroll))
  out("#endif\n");
  out("};\n\n");

# Generate <gensrc>/device_table.cpp
if is_colltrace:
  with open(os.path.join(gensrc, "device_table.cpp"), "w") as f:
    print("-- Generating %s" % os.path.join(gensrc, "device_table.cpp"))

    out = f.write
    out('#include "nccl_common.h"\n#include "device.h"\n')
    out("\n")

    seen_fns = set()
    out("const char* funcNames[FUNC_INDEX_TOTAL] = {\n")
    for fn in primary_funcs:
      fn_no_unroll = fn[:-1]
      if fn_no_unroll not in seen_fns:
        out('   "%s",\n' % paste("_", "ncclDevFunc", *fn_no_unroll))
        seen_fns.add(fn_no_unroll)
    for ty in all_tys:
      out(f'   "ncclDevFunc_OneRankReduce_PreMulSum_{ty}",\n')
    out("};\n")

# Generate <gensrc>/host_table.cpp
with open(os.path.join(gensrc, "host_table.cpp"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "host_table.cpp"))

  out = f.write
  out('#include "device.h"\n')
  out("\n")

  # The mapping from function rows to valid primary function ids.
  out("extern int const ncclDevFuncRowToId[] = {\n")
  index = 0
  offset = len(func_rows)//len(all_unroll)
  start = all_unroll.index(seen_unroll[0]) * offset
  end = start + offset
  for fn in func_rows[start:end]:
    fn_id, comment = -1, ""
    if fn is not None:
      fn_id = primary_to_index[equivalent_primary(*fn)] % offset if primary_to_index[equivalent_primary(*fn)] != -1 else -1
      comment = " // " + paste(" ", *fn[:-1])
    out("/*%4d*/ %d,%s\n" % (index, fn_id, comment))
    index += 1
  out(f"{index}")
  out("};\n")

# Maps to .cu filename which implements this func. The only constraint is that
# "coll" is reflected in the name: formally that no two funcs having different
# coll's map to the same filename.
def impl_filename(coll, algo, proto, redop, ty, unroll):
  return "%s.cpp" % paste("_", coll_camel_to_lower[coll], redop and redop.lower(), ty)

# Partition the functions and kernels to the .cu filenames. The partition is
# a dictionary mapping filename to (coll, func-tuple list)
def partition_by_name(fns):
  ans = {}
  for fn in fns:
    name = impl_filename(*fn)
    coll = fn[0]
    if name not in ans:
      ans[name] = (coll, [])
    ans[name][1].append(fn)
  return ans

name_to_funcs = partition_by_name(fn for fn in primary_funcs if fn[0]!="Nop")

redop_to_cxx = {
  None: "FuncCopy",
  "Sum": "FuncSum",
  "Prod": "FuncProd",
  "MinMax": "FuncMinMax",
  "PreMulSum": "FuncPreMulSum",
  "SumPostDiv": "FuncSumPostDiv"
}

ty_to_cxx = {
  None: "int8_t",
  "i8": "int8_t",
  "u8": "uint8_t",
  "i32": "int32_t",
  "u32": "uint32_t",
  "i64": "int64_t",
  "u64": "uint64_t",
  "f16": "half",
  "f32": "float",
  "f64": "double",
  "bf16": "hip_bfloat16",
  "f8e4m3":  "rccl_float8",
  "f8e5m2": "rccl_bfloat8"
}

# Generate each <gensrc>/<impl>.cpp:
for name in name_to_funcs.keys():
  (coll, fns) = name_to_funcs[name]
  with open(os.path.join(gensrc, name), "w") as f:
    print("-- Generating %s" % os.path.join(gensrc, name))

    out = f.write
    out(
      '#include "common.h"\n'
      '#include "{lower_coll}.h"\n'
      .format(lower_coll=coll_camel_to_lower[coll])
    )

    for fn in fns:
      (coll, algo, proto, redop, ty, unroll) = fn
      sym = paste("_", coll, algo, proto, redop, ty, unroll)
      if proto == "LL128":
        out("#if (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)) && defined(ENABLE_LL128)\n")
      out(
        "DEFINE_ncclDevFunc({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {unroll})\n"
        .format(sym=sym, coll=coll, redop_cxx=redop_to_cxx[redop], ty_cxx=ty_to_cxx[ty],
                algo=(algo or "RING"), proto=(proto or "SIMPLE"), unroll=unroll)
      )
      if proto == "LL128":
        out("#endif\n")

# Generate each <gensrc>/<msccl_impl>.cpp
if is_msccl_kernels:
  for redop in all_redops:
    if redop in ("Sum", "Prod", "MinMax"):
      for ty in all_tys:
        with open(os.path.join(gensrc, f"msccl_kernel_{redop}_{ty}.cpp"), "w") as f:
          print("-- Generating %s" % os.path.join(gensrc, f"msccl_kernel_{redop}_{ty}.cpp"))

          out = f.write
          out('#include "msccl_kernel_impl.h"\n#include "nccl_common.h"\n')
          out(
            "MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE({redop}, {ty_cxx}, false);\n"
            .format(redop=redop, ty_cxx=ty_to_cxx[ty])
          )
