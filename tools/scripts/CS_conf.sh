#!/bin/bash


##################################################### README #####################################################

# This script only requires 1 input arguement, it is the path to the rccl-tests repo. EX: /path/to/rccl-tests/

# Things to check before running this script, if things from the below list it is fine the script will still continue to
# run but it may produce an error on the missing parts.

# 1. UCX bin folder is on path or in default location under opt
# 2. OMPI bin folder is on path or in default location under opt
# 3. RCCL is either built and added to path or default loction is on path
# 4. RCCL-Tests are built
# 5. rocm-smi is on path
# 6. hipconfig is on path
# 7. rocminfo is on path
# 8. ibstatus is on path
# 9. ibv_devices is on path
# 10. ibv_devinfo is on path
# 11. ibstat device GUIDs

# All output will be in a folder called conf-script-output that will be created in the same directory as the script

##################################################### end README #####################################################



##################################################### define necessary functions #####################################################

function try() {
    local func_name="$1"
    local func_call="$2"
    local file_output="${func_name}_output.txt"
    local output

    # Shift to get past the label of the call/output file
    shift
    # Shift the arguments to pass the remaining ones to the called function
    shift

    # Run the command/function and capture its output
    output="$("${func_call}" "$@" 2>&1)"
    local exit_status=$?

    # Check if the command/function succeeded or failed
    if [ $exit_status -ne 0 ]; then
        catch "${func_call}" "${output}" "${func_name}"
    else
        echo "${output}" > "conf-script-output/${file_output}"
        echo "${func_name} was successful. Output saved to ${file_output}"
    fi
}

function catch() {
    local func_call="$1"
    local error_message="$2"
    local func_name="$3"
    echo "An error occurred during ${func_call}"
    echo "${error_message}"
    echo "in step ${func_name}"
}

# Function to get version info about ROCm
function rocmver()
{
    # Store the output of rocminfo in a variable
    rocminfo_output=$(rocminfo)

    # Grep the variable content for lines containing 'version'
    version_info=$(echo "$rocminfo_output" | grep -i "version")
    echo "$version_info"
}


# Function to get AMD GPU driver version
function amdgpuver()
{

    # Store the output of dkms in a variable
    dkms_output=$(dkms status)

    # Grep the variable content for lines containing 'amdgpu'
    amdgpu=$(echo "$dkms_output" | grep "amdgpu")
    echo "$amdgpu"

}

# Function to Query ACS
function ACSinfo()
{

    # Store the output of lspci in a variable
    lspci_output=$(lspci -vvv)

    # Grep the variable content for lines containing 'ACSCtl'
    acs=$(echo "$lspci_output" | grep ACSCtl)
    echo "$acs"

}

# Function to get rccl and rccl-tests version
run_rccl_tests() {

    local rccl_tests_dir="$1"

    # Get rccl-tests branch and version information
    local rccl_tests_branch=$(git -C "${rccl_tests_dir}" rev-parse --abbrev-ref HEAD)
    local rccl_tests_version=$(git -C "${rccl_tests_dir}" log -1 --format="%H")

    # Set the flag to display RCCL version during the run
    export NCCL_DEBUG=VERSION


    # Run the rccl-tests
    # Replace this line with the actual command to run rccl-tests in your environment

    local output_file="rccl_tests_output.txt"

    $1/build/all_reduce_perf -b 8 -e 16M -f 8 -g 2 > "${output_file}"

    # Unset the flag after execution
    unset NCCL_DEBUG

    # Extract RCCL, HIP, and ROCm versions from the output file
    local rccl_version=$(grep "RCCL version" "${output_file}" | awk '{print $4}')
    local hip_version=$(grep "HIP version" "${output_file}" | awk -F ': ' '{print $2}')
    local rocm_version=$(grep "ROCm version" "${output_file}" | awk -F ': ' '{print $2}')


    # Display extracted version information
    echo "RCCL Version: ${rccl_version}"
    echo "HIP Version: ${hip_version}"
    echo "ROCm Version: ${rocm_version}"

    # Display rccl-tests branch and version information
    echo "RCCL-Tests Branch: ${rccl_tests_branch}"
    echo "RCCL-Tests Version: ${rccl_tests_version}"
}

##################################################### end define necessary functions #####################################################



##################################################### setup output folder #####################################################

mkdir conf-script-output

##################################################### end setup output folder #####################################################



##################################################### query system with functions and commands for config info #####################################################

# ROCm version
try "ROCm_version" rocmver
echo ""

# GPU VRAM info
try "VRAM_info" rocm-smi --showmeminfo vram
echo ""

# HIP version
try "hip_version" hipconfig --version
echo ""

# RCCL version and RCCL tests version
try "RCCL_and_RCCL_tests_version" run_rccl_tests $1
echo ""

# UCX version
try "UCX_version" /opt/ucx/bin/ucx_info -v
echo ""

# MPI version4
try "MPI_version4" /opt/ompi4/bin/mpirun --version # the exact path might need to be removed in the context of debug
echo ""

# MPI version4
try "MPI_version5" /opt/ompi5/bin/mpirun --version
echo ""

# OS version
try "OS_version" cat /etc/os-release
echo ""

# Linux kernel version
try "Linux_Kernel_version" uname -r
echo ""

# ulimit -a
try "System_resource_allocation" ulimit -a
echo ""

# Environment Variable Config
try "Environment_Variable_Config" env
echo ""

# Rdma link info
try "rdma_link" rdma link
echo ""

# Query Numa balancing status
Try "Numa_Balancing" cat /proc/sys/kernel/numa_balancing
echo ""



# Infiniband device info
# IB device status
try "IB_device_status" ibstatus
echo ""

# IB device GUIDs
try "IB_devices" ibv_devices
echo ""

# IB device info
try "IB_devinfo" ibv_devinfo
echo ""

# IB device status alternate
try "IB_stat" ibstat
echo ""

# DKMS module info
try "dkms_status" dkms status
echo ""

# AMDKFD (GPU Driver version)
try "GPU_Driver_Version" amdgpuver
echo ""



# Network information
# IP addresses
try "IP_address_info" ip a
echo ""

# Network Interface state
try "IP_link_info" ip link
echo ""

# Route table info
try "IP_route_info" ip route
echo ""

# Access control service info
try "ACS_info" ACSinfo
echo ""

##################################################### end query system with functions and commands for config info #####################################################
