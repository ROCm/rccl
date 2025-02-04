#!/bin/bash


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
        echo "${output}" > "${file_output}"
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

# ROCm version
try "ROCm_version" rocmver
echo ""

# GPU VRAM info
try "VRAM_info" rocm-smi --showmeminfo vram
echo ""

# HIP version
try "hip_version" hipconfig --version
echo ""

# echo "6. RCCL version" ############################################ TO DO

# echo ""

# echo "7. RCCL-Tests version"

# echo "" ################################################################ END TO DO

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

# I think after I'm down I need to have all logs output to a folder, just a note to remind myself to do so