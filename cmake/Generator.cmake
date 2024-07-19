# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(ALL_PARAMS "ALL_COLLS" "ALL_ALGOS" "ALL_PROTOS" "ALL_REDOPS" "ALL_TYPES")
set(ALL_COLLS "AllGather" "AllReduce" "AllToAllPivot" "Broadcast" "Reduce" "ReduceScatter" "SendRecv")
set(ALL_ALGOS "TREE" "RING" "COLLNET_DIRECT" "COLLNET_CHAIN")
set(ALL_PROTOS "LL" "LL128" "SIMPLE")
set(ALL_REDOPS "Sum" "Prod" "MinMax" "PreMulSum" "SumPostDiv")
set(ALL_TYPES "int8_t" "uint8_t" "int32_t" "uint32_t" "int64_t" "uint64_t" "half" "float" "double" "hip_bfloat16" "rccl_float8" "rccl_bfloat8")

set(FLOATS_LIST "half" "float" "double" "hip_bfloat16" "rccl_float8" "rccl_bfloat8")

################################################################################
# The command line argument is used as a regex to filter the functions
# which make it into librccl. This is helpful for reducing the binary when
# developing device code. The regex supports non-space containing globs '*',
# and union 'a|b'. The string representing the function has the form:
#
# <coll> <algo> <proto> <redop> <type>
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
# # Only non-reductions:
# make ONLY_FUNCS="AllGather * *|Broadcast * *|SendRecv"
#
# # Only AllReduce Sum int32_t (but all algos, protos)
# make ONLY_FUNCS="AllReduce * * Sum int32_t"
#
# # Only AllReduce RING Max float (but all protos)
# make ONLY_FUNCS="AllReduce RING * Max float"
#
# # AllReduce TREE LL128 Prod rccl_bfloat16
# make ONLY_FUNCS="AllReduce TREE LL128 Prod rccl_bfloat16"
#
# # AllReduce RING SIMPLE and ReduceScatter RING LL float (but all redops, types for AllReduce and all redops for ReduceScatter)
# make ONLY_FUNCS="AllReduce RING SIMPLE * *|ReduceScatter RING LL * float"
#                         --- or ---
# make ONLY_FUNCS="AllReduce RING SIMPLE|ReduceScatter RING LL * float"
# make ONLY_FUNCS="AllReduce RING/TREE LL/SIMPLE Sum/MinMax int8_t/uint8_t/half/float/double/hip_bfloat16/rccl_float8/rccl_bfloat8|AllGather RING LL/SIMPLE Sum int8_t|AllToAllPivot RING SIMPLE Sum int8_t|Broadcast RING LL/SIMPLE Sum int8_t|Reduce RING LL/SIMPLE Sum/MinMax int8_t/uint8_t/half/float/double/hip_bfloat16/rccl_float8/rccl_bfloat8|ReduceScatter RING LL/SIMPLE Sum/MinMax int8_t/uint8_t/half/float/double/hip_bfloat16/rccl_float8/rccl_bfloat8|SendRecv RING SIMPLE Sum int8_t"

set(AllGather_Params     "RING" "*"      "Sum" "int8_t")
set(AllReduce_Params     "*"    "*"      "*"   "*")
set(AllToAllPivot_Params "RING" "SIMPLE" "Sum" "int8_t")
set(Broadcast_Params     "RING" "*"      "Sum" "int8_t")
set(Reduce_Params        "RING" "*"      "*"   "*")
set(ReduceScatter_Params "RING" "*"      "*"   "*")
set(SendRecv_Params      "RING" "SIMPLE" "Sum" "int8_t")

#############################################################################################################
## Helper function to check if the conditions for the collective is being met
#############################################################################################################
function(validate_func ITEM_LIST)
  set(paramIdx 1)
  ## Extract coll/redop/type
  list(GET ITEM_LIST 0 coll)
  list(GET ITEM_LIST 3 redop)
  list(GET ITEM_LIST 4 type)

  ## First check if redop 'SumPostDiv' has no type float
  if(${redop} STREQUAL "SumPostDiv" AND type IN_LIST FLOATS_LIST)
    set(is_valid FALSE PARENT_SCOPE)
    return()
  endif()
  foreach(parameter IN LISTS "${coll}_Params")
    if(NOT parameter STREQUAL "*")
      list(GET ITEM_LIST "${paramIdx}" item)
      string(FIND "${parameter}" "${item}" is_found)
      if(is_found EQUAL -1)
        set(is_valid FALSE PARENT_SCOPE)
        return()
      endif()
    endif()
    math(EXPR paramIdx "${paramIdx} + 1")
  endforeach()
  set(is_valid TRUE PARENT_SCOPE)
endfunction()

#############################################################################################################
## A recursive helper macro to generate functions and kernels based on the input given
#############################################################################################################
macro(filter_functions FUNCTION_PARAMS current_idx)
  ## Check if the current_idx does not exceed the max depth
  if(${current_idx} LESS 5)
    ## current_element is the config parameter
    list(GET FUNCTION_PARAMS ${current_idx} current_element)

    ## If the parameter is equal to '*', include all the possible cases for it
    if(${current_element} STREQUAL "*")
      if(${current_idx} EQUAL 0)
        message(FATAL_ERROR "Error: Parameter 'COLL' can not be type all '*'.")
      endif()
      ## ALL_PARAMS list must be in the same order as FUNCTION_PARAMS ---> <coll> <algo> <proto> <redop> <type>
      ## Find the respective parameter list from ALL_PARAMS list
      list(GET ALL_PARAMS ${current_idx} current_list)

      ## Iterate over the items int the current_list
      foreach(item IN LISTS ${current_list})
        ## Add item to ITEM_LIST which will be used in the inner most loop
        list(APPEND ITEM_LIST ${item})
        math(EXPR new_idx "${current_idx} + 1")
        filter_functions(${FUNCTION_PARAMS} ${new_idx} ${ARGN})

        ## For each loop layer remove the last element in ITEM_LIST
        list(REMOVE_AT ITEM_LIST -1)
      endforeach()
    else()
      ## Check if the current element is recognized
      list(GET ALL_PARAMS ${current_idx} current_param)
      string(REPLACE "/" ";" elements ${current_element})
      ## Iterate over the elements int the ELEMENTS_LIST
      foreach(item IN LISTS elements)
        list(FIND ${current_param} ${item} is_valid)
        if(${is_valid} EQUAL -1)
          message(FATAL_ERROR "Error: ${item} is unrecognized or does not belong to this category ${current_param}.")
        endif()
      endforeach()
      foreach(item IN LISTS elements)
        ## Add item to ITEM_LIST which will be used in the inner most loop
        list(APPEND ITEM_LIST ${item})
        math(EXPR new_idx "${current_idx} + 1")
        filter_functions(${FUNCTION_PARAMS} ${new_idx} ${ARGN})

        ## For each loop layer remove the last element in ITEM_LIST
        list(REMOVE_AT ITEM_LIST -1)
      endforeach()
    endif()
  else()
    ## This is the inner most loop where the file is generated
    ## Unzip ITEM_LIST
    list(GET ITEM_LIST 0 COLL)
    list(GET ITEM_LIST 1 ALGO)
    list(GET ITEM_LIST 2 PROTO)
    list(GET ITEM_LIST 3 REDOP)
    list(GET ITEM_LIST 4 TYPE)

    validate_func("${ITEM_LIST}")
    if (NOT is_valid)
      continue()
    endif()

    list(APPEND COLL_LIST "${COLL}-${ALGO}-${PROTO}-${REDOP}-${TYPE}")
    set(COLL_LIST ${COLL_LIST} PARENT_SCOPE)

    ## Append the newly formed function/kernel to list
    list(APPEND FUNC_LIST "ncclDevFunc_${COLL}_${ALGO}_${PROTO}_${REDOP}_${TYPE}")
    list(APPEND KERN_LIST "ncclDevKernel_${COLL}_${ALGO}_${PROTO}_${REDOP}_${TYPE}")
    set(FUNC_LIST ${FUNC_LIST} PARENT_SCOPE)
    set(KERN_LIST ${KERN_LIST} PARENT_SCOPE)
  endif()
endmacro()

#####################################################################################################
## Function to generate device table
#####################################################################################################
function(gen_device_table)
  ## Generate device table and list all the functions
  set(DEVICE_TABLE_H_FILE "${HIPIFY_DIR}/src/device/device_table.h")
  message(STATUS "Generating ${DEVICE_TABLE_H_FILE}")

  if(ENABLE_IFC)
    set(func_declaration "__device__ void")
  else()
    set(func_declaration "__device__ __attribute__((noinline)) void")
  endif()

  ## Declaration of device functions
  foreach(func IN LISTS FUNC_LIST)
    string(FIND "${func}" "LL128" IS_LL128)
    if(NOT IS_LL128 EQUAL -1)
      file(APPEND ${DEVICE_TABLE_H_FILE} "#if defined(__gfx90a__) && defined(ENABLE_LL128)\n")
      file(APPEND ${DEVICE_TABLE_H_FILE} "${func_declaration} ${func}();\n${func_declaration} ${func}_4();\n#else\n")
      string(REPLACE "LL128" "LL" func "${func}")
      file(APPEND ${DEVICE_TABLE_H_FILE} "${func_declaration} ${func}();\n${func_declaration} ${func}_4();\n#endif\n")
    else()
      file(APPEND ${DEVICE_TABLE_H_FILE} "${func_declaration} ${func}();\n${func_declaration} ${func}_4();\n")
    endif()
  endforeach()
  file(APPEND ${DEVICE_TABLE_H_FILE} "\n")

  ## Undirect function call
  file(APPEND ${DEVICE_TABLE_H_FILE} "typedef void(*ncclDevFuncPtr_t)();\n\n")
  file(APPEND ${DEVICE_TABLE_H_FILE} "__device__ ncclDevFuncPtr_t const ncclDevFuncTable[] = {\n")
  foreach(func ${FUNC_LIST})
    string(FIND "${func}" "LL128" IS_LL128)
    if(NOT IS_LL128 EQUAL -1)
      file(APPEND ${DEVICE_TABLE_H_FILE} "#if defined(__gfx90a__) && defined(ENABLE_LL128)\n")
      file(APPEND ${DEVICE_TABLE_H_FILE} "  ${func},\n#else\n")
      string(REPLACE "LL128" "LL" func "${func}")
      file(APPEND ${DEVICE_TABLE_H_FILE} "  ${func},\n#endif\n")
    else()
      file(APPEND ${DEVICE_TABLE_H_FILE} "  ${func},\n")
    endif()
  endforeach()
  file(APPEND ${DEVICE_TABLE_H_FILE} "nullptr};\n\n")
  file(APPEND ${DEVICE_TABLE_H_FILE} "__device__ ncclDevFuncPtr_t const ncclDevFuncTable_4[] = {\n")
  foreach(func ${FUNC_LIST})
    string(FIND "${func}" "LL128" IS_LL128)
    if(NOT IS_LL128 EQUAL -1)
      file(APPEND ${DEVICE_TABLE_H_FILE} "#if defined(__gfx90a__) && defined(ENABLE_LL128)\n")
      file(APPEND ${DEVICE_TABLE_H_FILE} "  ${func}_4,\n#else\n")
      string(REPLACE "LL128" "LL" func "${func}")
      file(APPEND ${DEVICE_TABLE_H_FILE} "  ${func}_4,\n#endif\n")
    else()
      file(APPEND ${DEVICE_TABLE_H_FILE} "  ${func}_4,\n")
    endif()
  endforeach()
  file(APPEND ${DEVICE_TABLE_H_FILE} "nullptr};\n\n")

  if(NOT ENABLE_IFC)
    ## Direct functions calls
    file(APPEND ${DEVICE_TABLE_H_FILE}
      "template<unsigned short f, unsigned short l>\n"
      "struct Caller {\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call(unsigned short funcIndex) noexcept\n"
      "  {\n"
      "    constexpr unsigned short m = f + (l - f) / 2;\n"
      "    return (funcIndex < m) ? Caller<f, m>::call(funcIndex) : Caller<m, l>::call(funcIndex);\n"
      "  }\n"
      "};\n"
      "\n"
      "template<unsigned short f>\n"
      "struct Caller<f, f + 1>{\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call(unsigned short funcIndex) noexcept { ncclDevFuncTable[f](); }\n"
      "};\n"
    )
    file(APPEND ${DEVICE_TABLE_H_FILE} "__forceinline__ __device__ void NCCL_CALL_FUNCTIONS(unsigned short funcIndex) noexcept {\n")
    list(LENGTH FUNC_LIST max_index)
    file(APPEND ${DEVICE_TABLE_H_FILE} "  Caller<0, ${max_index}>::call(funcIndex);\n}\n\n")

    file(APPEND ${DEVICE_TABLE_H_FILE}
      "template<unsigned short f, unsigned short l>\n"
      "struct Caller4 {\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call4(unsigned short funcIndex) noexcept\n"
      "  {\n"
      "    constexpr unsigned short m = f + (l - f) / 2;\n"
      "    return (funcIndex < m) ? Caller4<f, m>::call4(funcIndex) : Caller4<m, l>::call4(funcIndex);\n"
      "  }\n"
      "};\n"
      "\n"
      "template<unsigned short f>\n"
      "struct Caller4<f, f + 1>{\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call4(unsigned short funcIndex) noexcept { ncclDevFuncTable_4[f](); }\n"
      "};\n"
    )
    file(APPEND ${DEVICE_TABLE_H_FILE} "__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_4(unsigned short funcIndex) noexcept {\n")
    list(LENGTH FUNC_LIST max_index)
    file(APPEND ${DEVICE_TABLE_H_FILE} "  Caller4<0, ${max_index}>::call4(funcIndex);\n}\n\n")
  endif()

  ## Function name table for collective trace
  if(COLLTRACE)
    set(DEVICE_TABLE_FILE "${HIPIFY_DIR}/src/device/device_table.cpp")
    message(STATUS "Generating ${DEVICE_TABLE_FILE}")

    file(APPEND ${DEVICE_TABLE_FILE} "#include \"nccl_common.h\"\n#include \"device.h\"\n\n const char* funcNames[FUNC_INDEX_TOTAL] = {\n")
    foreach(func ${FUNC_LIST})
      file(APPEND ${DEVICE_TABLE_FILE} "  \"${func}\",\n")
    endforeach()
    foreach(type IN LISTS ALL_TYPES)
      file(APPEND ${DEVICE_TABLE_FILE} "  \"ncclDevFunc_OneRankReduce_PreMulSum_${type}\",\n")
    endforeach()
    file(APPEND ${DEVICE_TABLE_FILE} "};\n")
  endif()

  ## Add the device_table files to HIP_SOURCES
  list(APPEND HIP_SOURCES ${DEVICE_TABLE_H_FILE})
  list(APPEND HIP_SOURCES ${DEVICE_TABLE_FILE})
  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
endfunction()

######################################################################################################
## Function to generate host-side table
######################################################################################################
function(gen_host_table)
  set(HOST_TABLE_FILE "${HIPIFY_DIR}/src/device/host_table.cpp")
  message(STATUS "Generating ${HOST_TABLE_FILE}")

  file(WRITE ${HOST_TABLE_FILE} "#include \"device.h\"\n\n")

  ## The mapping from function rows to valid function ids
  file(APPEND ${HOST_TABLE_FILE} "extern int const ncclDevFuncRowToId[] = {\n")
  set(idx 0)
  foreach(coll IN LISTS ALL_COLLS)
    foreach(algo IN LISTS ALL_ALGOS)
      foreach(proto IN LISTS ALL_PROTOS)
        foreach(redop IN LISTS ALL_REDOPS)
          foreach(type IN LISTS ALL_TYPES)
            ## Create a list from the combination of curr parameters
            set(ITEM_LIST ${coll} ${algo} ${proto} ${redop} ${type})
            validate_func("${ITEM_LIST}")
            if (NOT is_valid)
              continue()
            endif()
            ## Try to find the combination in the generated func list
            list(FIND FUNC_LIST "ncclDevFunc_${coll}_${algo}_${proto}_${redop}_${type}" fn_id)
            if(NOT ${fn_id} EQUAL -1)
            set(last_valid_fn_id ${fn_id})
              file(APPEND ${HOST_TABLE_FILE} "  /*${idx}*/ ${fn_id}, // ncclDevFunc_${coll}_${algo}_${proto}_${redop}_${type}\n")
            else()
              file(APPEND ${HOST_TABLE_FILE} "  /*${idx}*/ ${fn_id},\n")
            endif()
            math(EXPR idx "${idx} + 1")
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  math(EXPR last_valid_fn_id "${last_valid_fn_id} + 1")
  file(APPEND ${HOST_TABLE_FILE} "${last_valid_fn_id}};\n\n")

  ## Add the host_table file to HIP_SOURCES
  list(APPEND HIP_SOURCES ${HOST_TABLE_FILE})
  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
endfunction()

###########################################################################################################
## Function to generate MSCCL Kernels
###########################################################################################################
function(gen_msccl_kernels)
  set(MSCCL_REDOP Sum Prod MinMax)
  foreach(REDOP_CURRENT IN LISTS MSCCL_REDOP)
    foreach(DATA_TYPE ${ALL_TYPES})
      set(FILE_NAME "${HIPIFY_DIR}/src/device/msccl_kernel_${REDOP_CURRENT}_${DATA_TYPE}.cpp")
      message(STATUS "Generating ${FILE_NAME}")
      file(WRITE ${FILE_NAME}
        "#include \"msccl_kernel_impl.h\"
        #include \"nccl_common.h\"
        MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(${REDOP_CURRENT}, ${DATA_TYPE}, false);")
      list(APPEND HIP_SOURCES ${FILE_NAME})
    endforeach()
  endforeach()
  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
endfunction()

###########################################################################################################
## Function to generate collectives
###########################################################################################################
function(gen_collectives)
  # Iterate over each item in the original list
  foreach(item ${COLL_LIST})
    # Split the string into components
    string(REPLACE "-" ";" item_components ${item})

    # Extract COLL, ALGO, and PROTO components
    list(GET item_components 0 coll_prefix)
    list(GET item_components 1 algo_prefix)
    list(GET item_components 2 proto_prefix)
    list(GET item_components 3 redop_prefix)

    # Create a list name using COLL, ALGO, and PROTO
    set(list_name "${coll_prefix}_${algo_prefix}_${proto_prefix}_${redop_prefix}")

    # Add the item to the corresponding list
    list(APPEND ${list_name} ${item})

    # Add the list name to the map if it doesn't exist
    if(NOT list_name IN_LIST divided_lists)
      list(APPEND divided_lists ${list_name})
    endif()
  endforeach()

  set(index 0)
  foreach(list_name IN LISTS divided_lists)
    foreach(item IN LISTS ${list_name})
        # Convert to a list
        string(REPLACE "-" ";" components ${item})

        list(GET components 0 coll)
        list(GET components 1 algo)
        list(GET components 2 proto)
        list(GET components 3 redop)
        list(GET components 4 type)

        list(APPEND IMPL_LIST "DEFINE_ncclDevFunc(${coll}_${algo}_${proto}_${redop}_${type}, ncclFunc${coll}, Func${redop}, ${type}, NCCL_ALGO_${algo}, NCCL_PROTO_${proto})\n")

        # Increment the function id
        math(EXPR index "${index} + 1")
    endforeach()
    ## Store lower-case version of COLL
    string(TOLOWER ${coll} COLL_LOWER)
    string(REPLACE "scatter" "_scatter" COLL_LOWER ${COLL_LOWER})
    if(NOT ${coll} STREQUAL "AllToAllPivot")
      string(REPLACE "all" "all_" COLL_LOWER ${COLL_LOWER})
    else()
      string(REPLACE "pivot" "_pivot" COLL_LOWER ${COLL_LOWER})
    endif()

    ## Set name/path of the file
    set(FILE_PATH "${HIPIFY_DIR}/src/device/${list_name}.cpp")
    message(STATUS "Generating ${FILE_PATH}")

    ## Construct the file
    file(WRITE ${FILE_PATH} "#include \"${COLL_LOWER}.h\"\n#include \"common.h\"\n\n")
    string(FIND "${list_name}" "LL128" IS_LL128)
    if(NOT IS_LL128 EQUAL -1)
      file(APPEND ${FILE_PATH} "#if defined(__gfx90a__) && defined(ENABLE_LL128)\n")
    endif()
    foreach(IMPL IN LISTS IMPL_LIST)
      file(APPEND ${FILE_PATH} "${IMPL}")
    endforeach()
    if(NOT IS_LL128 EQUAL -1)
      file(APPEND ${FILE_PATH} "#endif\n")
    endif()

    ## Append the file to HIP sources list which will be added to source list
    list(APPEND HIP_SOURCES ${FILE_PATH})
    set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)

    # Clear the IMPL list for the next iteration
    set(IMPL_LIST)
  endforeach()
endfunction()

###################################################################################################################
## Function to generate all the functions that are going to be in librccl.so
###################################################################################################################
function(gen_functions CONFIG_INPUT)
  string(REPLACE "|" ";" INPUT_LIST ${CONFIG_INPUT})
  ## Sort the input so that it matches ALL_COLLS
  list(SORT INPUT_LIST)

  foreach(INPUT ${INPUT_LIST})
    # Parse the the config string and make it a list
    string(REPLACE " " ";" FUNCTION_PARAMS ${INPUT})

    # Get the number of parameters in the input
    list(LENGTH FUNCTION_PARAMS PARAMS_LENGTH)

    # Assume all if a parameter is missing
    while(${PARAMS_LENGTH} LESS 5)
      list(APPEND FUNCTION_PARAMS "*")
      list(LENGTH FUNCTION_PARAMS PARAMS_LENGTH)
    endwhile()

    ## Filter functions/kernels based on input
    filter_functions(FUNCTION_PARAMS 0)
  endforeach()

  gen_collectives()     ## Generate collective files
  if(ENABLE_MSCCL_KERNEL)
    gen_msccl_kernels() ## Generate msccl files (not configurable)
  endif()
  gen_device_table()    ## Generate device_table.cpp
  gen_host_table()      ## Generate host_table.cpp

  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
endfunction()
