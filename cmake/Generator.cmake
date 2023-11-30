# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
set(ALL_REDOPS "Sum" "Prod" "Max" "Min" "PreMulSum" "SumPostDiv")
set(ALL_TYPES "int8_t" "uint8_t" "int32_t" "uint32_t" "int64_t" "uint64_t" "half" "float" "double" "rccl_bfloat16")

set(FLOATS_LIST "half" "float" "double" "rccl_bfloat16")

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
      list(FIND ${current_param} ${current_element} is_valid)
      if(${is_valid} EQUAL -1)
        message(FATAL_ERROR "Error: ${current_element} is unrecognized or does not belong to this category.")
      endif()

      ## If not '*', no need to iterate. Add the current_element to ITEM_LIST
      list(APPEND ITEM_LIST ${current_element})
      math(EXPR new_idx "${current_idx} + 1")
      filter_functions(${FUNCTION_PARAMS} ${new_idx} ${ARGN})

      list(REMOVE_AT ITEM_LIST -1)
    endif()
  else()
    ## This is the inner most loop where the file is generated
    ## Unzip ITEM_LIST
    list(GET ITEM_LIST 0 COLL)
    list(GET ITEM_LIST 1 ALGO)
    list(GET ITEM_LIST 2 PROTO)
    list(GET ITEM_LIST 3 REDOP)
    list(GET ITEM_LIST 4 TYPE)

    ## Need to check if these conditions are met prior to file generation
    if(NOT ${COLL} STREQUAL "AllReduce" AND NOT ${ALGO} STREQUAL "RING")
      continue()
    elseif((${COLL} STREQUAL "AllGather" OR ${COLL} STREQUAL "Broadcast" OR ${COLL} STREQUAL "SendRecv" OR ${COLL} STREQUAL "AllToAllPivot") AND (NOT ${REDOP} STREQUAL "Sum" OR NOT ${TYPE} STREQUAL "int8_t"))
      continue()
    elseif((${COLL} STREQUAL "SendRecv" OR ${COLL} STREQUAL "AllToAllPivot") AND NOT ${PROTO} STREQUAL "SIMPLE")
      continue()
    endif()

    if(${REDOP} STREQUAL "SumPostDiv" AND TYPE IN_LIST FLOATS_LIST)
      continue()
    endif()

    list(APPEND COLL_LIST "${COLL}-${ALGO}-${PROTO}-${REDOP}-${TYPE}")
    set(COLL_LIST ${COLL_LIST} PARENT_SCOPE)

    ## Append the newly formed function/kernel to list
    list(APPEND FUNC_LIST "ncclFunction_${COLL}_${ALGO}_${PROTO}_${REDOP}_${TYPE}")
    list(APPEND KERN_LIST "ncclKernel_${COLL}_${ALGO}_${PROTO}_${REDOP}_${TYPE}")
    set(FUNC_LIST ${FUNC_LIST} PARENT_SCOPE)
    set(KERN_LIST ${KERN_LIST} PARENT_SCOPE)
  endif()
endmacro()

#####################################################################################################
## Function to generate device table
#####################################################################################################
function(gen_device_table)
  set(DEVICE_TABLE_FILE "${HIPIFY_DIR}/src/collectives/device/device_table.cpp")
  message(STATUS "Generating ${DEVICE_TABLE_FILE}")

  ## Generate device table and list all the functions
  file(WRITE ${DEVICE_TABLE_FILE} "#include \"common.h\"\n#include \"collectives.h\"\n\n")

  ## Declaration of device functions
  foreach(func IN LISTS FUNC_LIST)
    if(ENABLE_IFC)
      file(APPEND ${DEVICE_TABLE_FILE} "__device__ void ${func}();\n")
    else()
      file(APPEND ${DEVICE_TABLE_FILE} "__device__ __attribute__((noinline)) void ${func}();\n")
    endif()
  endforeach()
  file(APPEND ${DEVICE_TABLE_FILE} "\n")

  if(ENABLE_IFC)
    ## Undirect function call
    file(APPEND ${DEVICE_TABLE_FILE} "__device__ ncclKernelFunc_t const ncclFuncs[] = {\n")
    foreach(func ${FUNC_LIST})
      file(APPEND ${DEVICE_TABLE_FILE} "  ${func},\n")
    endforeach()
    ## Add OneRankReduce functions at the end
    foreach(type IN LISTS ALL_TYPES)
      file(APPEND ${DEVICE_TABLE_FILE} "  ncclFunction_OneRankReduce_PreMulSum_${type},\n")
    endforeach()
    file(APPEND ${DEVICE_TABLE_FILE} "nullptr};\n\n")
  else()
    ## Direct functions calls
    file(APPEND ${DEVICE_TABLE_FILE} "__device__ void NCCL_CALL_FUNCTIONS(unsigned short funcIndex) noexcept {\n  switch(funcIndex) {\n")
    set(index 0)
    foreach(func IN LISTS FUNC_LIST)
      file(APPEND ${DEVICE_TABLE_FILE} "    case ${index}:\n      ${func}();\n      break;\n")
      math(EXPR index "${index} + 1")
    endforeach()
    ## Add OneRankReduce functions at the end
    foreach(type IN LISTS ALL_TYPES)
      file(APPEND ${DEVICE_TABLE_FILE} "    case ${index}:\n      ncclFunction_OneRankReduce_PreMulSum_${type}();\n      break;\n")
      math(EXPR index "${index} + 1")
    endforeach()
    file(APPEND ${DEVICE_TABLE_FILE} "  }\n}\n")
  endif()

  ## Add the device_table file to HIP_SOURCES
  list(APPEND HIP_SOURCES ${DEVICE_TABLE_FILE})
  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
endfunction()

######################################################################################################
## Function to generate host-side table
######################################################################################################
function(gen_host_table)
  set(HOST_TABLE_FILE "${HIPIFY_DIR}/src/collectives/device/host_table.cpp")
  message(STATUS "Generating ${HOST_TABLE_FILE}")

  file(WRITE ${HOST_TABLE_FILE} "#include \"devcomm.h\"\n\n")

  ## The mapping from function rows to valid function ids
  file(APPEND ${HOST_TABLE_FILE} "extern int const ncclFuncRowToId[] = {\n")
  set(idx 0)
  foreach(coll IN LISTS ALL_COLLS)
    foreach(algo IN LISTS ALL_ALGOS)
      foreach(proto IN LISTS ALL_PROTOS)
        foreach(redop IN LISTS ALL_REDOPS)
          foreach(type IN LISTS ALL_TYPES)
            if(NOT ${coll} STREQUAL "AllReduce" AND NOT ${algo} STREQUAL "RING")
              continue()
            elseif((${coll} STREQUAL "AllGather" OR ${coll} STREQUAL "Broadcast" OR ${coll} STREQUAL "SendRecv" OR ${coll} STREQUAL "AllToAllPivot") AND (NOT ${redop} STREQUAL "Sum" OR NOT ${type} STREQUAL "int8_t"))
              continue()
            elseif((${coll} STREQUAL "SendRecv" OR ${coll} STREQUAL "AllToAllPivot") AND NOT ${proto} STREQUAL "SIMPLE")
              continue()
            endif()

            if(${redop} STREQUAL "SumPostDiv" AND type IN_LIST FLOATS_LIST)
              continue()
            endif()

            list(FIND FUNC_LIST "ncclFunction_${coll}_${algo}_${proto}_${redop}_${type}" fn_id)
            if(NOT ${fn_id} EQUAL -1)
              file(APPEND ${HOST_TABLE_FILE} "  /*${idx}*/ ${fn_id}, // ncclFunction_${coll}_${algo}_${proto}_${redop}_${type}\n")
            else()
              file(APPEND ${HOST_TABLE_FILE} "  /*${idx}*/ ${fn_id},\n")
            endif()
            math(EXPR idx "${idx} + 1")
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  math(EXPR fn_id "${fn_id} + 1")
  ## Add OneRankReduce function ids at the end
  foreach(type IN LISTS ALL_TYPES)
    file(APPEND ${HOST_TABLE_FILE} "  /*${idx}*/ ${fn_id}, // ncclFunction_OneRankReduce_PreMulSum_${type}\n")

    ## Increment the index and func id for each OneRankReduce
    math(EXPR idx "${idx} + 1")
    math(EXPR fn_id "${fn_id} + 1")
  endforeach()
  file(APPEND ${HOST_TABLE_FILE} "-1};\n\n")

  ## Add the host_table file to HIP_SOURCES
  list(APPEND HIP_SOURCES ${HOST_TABLE_FILE})
  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
endfunction()

###########################################################################################################
## Function to generate MSCCL Kernels
###########################################################################################################
function(gen_msccl_kernels)
  set(REDOP_FILTERED Sum Prod Min Max PreMulSum SumPostDiv)
  foreach(REDOP_CURRENT IN LISTS REDOP_FILTERED)
    foreach(DATA_TYPE ${ALL_TYPES})
      set(FILE_NAME "${HIPIFY_DIR}/src/collectives/device/msccl_kernel_${REDOP_CURRENT}_${DATA_TYPE}.cpp")
      message(STATUS "Generating ${FILE_NAME}")
      file(WRITE ${FILE_NAME}
        "#include \"msccl_kernel_impl.h\"
        #include \"primitives.h\"
        #include \"collectives.h\"
        #include \"devcomm.h\"
        MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(${REDOP_CURRENT}, ${DATA_TYPE}, false);
        MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(${REDOP_CURRENT}, ${DATA_TYPE}, true);")
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

        list(APPEND IMPL_LIST "IMPL_COLL_FUNC(${coll}, ${algo}, ${proto}, ${redop}, ${type})\n")

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
    set(FILE_PATH "${HIPIFY_DIR}/src/collectives/device/${list_name}.cpp")
    message(STATUS "Generating ${FILE_PATH}")

    ## Construct the file
    file(WRITE ${FILE_PATH} "#include \"${COLL_LOWER}.h\"\n#include \"common.h\"\n#include \"collectives.h\"\n")
    foreach(IMPL IN LISTS IMPL_LIST)
      file(APPEND ${FILE_PATH} "${IMPL}")
    endforeach()

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