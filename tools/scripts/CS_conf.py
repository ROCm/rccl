import subprocess
import time
import os
import re


class CommandResult:
    def __init__(self, stdout, stderr):
        self.stdout = stdout
        self.stderr = stderr

# Function to center the titles in the detailed section
def centered_title(title, width, fill_char=" "):
    padding_width = (width - len(title)) // 2
    return f'{fill_char*padding_width}{title}{fill_char*padding_width}\n'


# Function to run a CLI command and return its output
def run_cli_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Get the status of a particular command
def status_check(summary, result):
    # List of errors to check
    error_list = [r'No such file or directory', r'Command not found', r'Permission denied', r'cannot access', r'error']
    status = "OK"
    if summary == "Missing Data":
        status = "WARN"
    for error in error_list:
        match = re.search(error, result.stderr, re.IGNORECASE)
        if match:
            status = "WARN"
            break
    return status

# Check if a directory is on path or LD_LIBRARY_PATH
def PATH_and_LD_LIBRARY_PATH(dir):
    try:
        path = os.environ.get('PATH')
        LD_path = os.environ.get('LD_LIBRARY_PATH')
    except Exception as e:
        return False
    pattern = re.escape(dir)
    match_path = re.search(pattern, path)
    match_LD_path = re.search(pattern, LD_path)
    if match_LD_path and match_path:
        return True
    return False


# Get OS version
def get_os_version():
    result = run_cli_command('cat /etc/os-release')
    match = re.search(r'PRETTY_NAME="(.+)"', result.stdout)
    if match:
        summary = match.group(1)
    else:
        summary = "Missing Data"
    return summary, result

# Get ROCm Version
def get_ROCm_version():
    result = run_cli_command('cat /opt/rocm/.info/version')
    if result.stdout:
        summary = result.stdout.strip()
    else:
        summary = "Missing Data"
    return summary, result

# Get HIP Version
def get_HIP_version():
    result = run_cli_command('hipconfig --version')
    if result.stdout:
        summary = result.stdout.strip()
    else:
        summary = "Missing Data"
    return summary, result

# Get Vram Information
def get_Vram_info():
    result = run_cli_command('rocm-smi --showmeminfo vram')
    if result.stdout:
        summary = "Memory Usage is detailed in the Vram Information section"
    else:
        summary = "Missing Data"
    return summary, result

# Get UCX version
def ucx_version():
    path_check = PATH_and_LD_LIBRARY_PATH(dir="ucx")
    if path_check:
        result = run_cli_command('ucx_info -v')
        match = re.search(r"Library version: (\d+\.\d+\.\d+)", result.stdout)
        if match:
            summary = match.group(1)
        else:
            summary = "Missing Data"
        return summary, result
    else:
        stdout = ""
        stderr = "Error: UCX not on PATH or LD_LIBRARY_PATH"
        result = CommandResult(stdout=stdout,stderr=stderr)
        summary = "UCX not on PATH or LD_LIBRARY_PATH"
        return summary, result

# Get MPI version
def mpi_version():
    path_check = PATH_and_LD_LIBRARY_PATH(dir="ompi")
    if path_check:
        result = run_cli_command('mpirun --version')
        match = re.search(r"mpirun \(Open MPI\) \d+\.\d+\.\d+", result.stdout)
        if match:
            summary = match.group()
        else:
            summary = "Missing Data"
        return summary, result
    else:
        stdout = ""
        stderr = "Error: ompi4 or ompi5 (only 1 is required) not on PATH or LD_LIBRARY_PATH"
        result = CommandResult(stdout=stdout,stderr=stderr)
        summary = "ompi4 or ompi5 (only 1 is required) not on PATH or LD_LIBRARY_PATH"
        return summary, result

# Get Linux kernel version
def get_Linux_kernel_version():
    result = run_cli_command('uname -r')
    if result.stdout:
        summary = result.stdout.strip()
    else:
        summary = "Missing Data"
    return summary, result

# Get Resource limits
def get_resource_limits_info():
    result = run_cli_command('ulimit -a')
    if result.stdout:
        summary = "Output is detailed in the Resource limits section"
    else:
        summary = "Missing Data"
    return summary, result

# Get Environment config
def get_Environment_config_info():
    result = run_cli_command('env')
    if result.stdout:
        summary = "Output is detailed in the Environment Config section"
    else:
        summary = "Missing Data"
    return summary, result

# Get Rdma link info
def get_rdma_link_info():
    result = run_cli_command('rdma link')
    if result.stdout:
        summary = "Output is detailed in the rdma link section"
    else:
        summary = "Missing Data"
    return summary, result

# Get NUMA Balancing
def get_NUMA_balancing_info():
    result = run_cli_command('cat /proc/sys/kernel/numa_balancing')
    if result.stdout:
        summary = result.stdout 
    else:
        summary = "Missing Data"
    return summary, result


# Gather all data and build summary table and detailed output format
def get_config():
    # Run the commands and store the command outputs

    # OS version
    os_summary, os_result = get_os_version()
    os_status = status_check(os_summary, os_result)

    # ROCm Version
    ROCm_summary, ROCm_result = get_ROCm_version()
    ROCm_status = status_check(ROCm_summary, ROCm_result)

    # HIP Version
    HIP_summary, HIP_result = get_HIP_version()
    HIP_status = status_check(HIP_summary, HIP_result)

    # Vram info
    Vram_summary, Vram_result = get_Vram_info()
    Vram_status = status_check(Vram_summary, Vram_result)

    # UCX Version
    ucx_summary, ucx_result = ucx_version()
    ucx_status = status_check(ucx_summary, ucx_result)

    # MPI Version
    mpi_summary, mpi_result = mpi_version()
    mpi_status = status_check(mpi_summary, mpi_result)

    # Linux kernel version
    Lkv_summary, Lkv_result = get_Linux_kernel_version()
    Lkv_status = status_check(Lkv_summary, Lkv_result)

    # Resource limits
    rlv_summary, rlv_result = get_resource_limits_info()
    rlv_status = status_check(rlv_summary, rlv_result)

    # Environment config
    env_summary, env_result = get_Environment_config_info()
    env_status = status_check(env_summary, env_result)

    # Rdma link info
    rdl_summary, rdl_result = get_rdma_link_info()
    rdl_status = status_check(rdl_summary, rdl_result)
    
    # NUMA Balancing info
    nb_summary, nb_result = get_NUMA_balancing_info()
    nb_status = status_check(nb_summary, nb_result)

    # Create the summary table
    summary_table = (
        f"\n\n{'='*119}\n"
        f"{'Component':<30}| {'Status':<13} | Value\n"
        f"{'='*119}\n"
        f"OS Version{' ':<20}| {os_status:<13} | {os_summary}\n"
        f"ROCm Version{' ':<18}| {ROCm_status:<13} | {ROCm_summary}\n"
        f"HIP Version{' ':<19}| {HIP_status:<13} | {HIP_summary}\n"
        f"Vram Information{' ':<14}| {Vram_status:<13} | {Vram_summary}\n"
        f"UCX Version{' ':<19}| {ucx_status:<13} | {ucx_summary}\n"
        f"MPI Version{' ':<19}| {mpi_status:<13} | {mpi_summary}\n"
        f"Linux Kernel Version{' ':<10}| {Lkv_status:<13} | {Lkv_summary}\n"
        f"Resource limits{' ':<15}| {rlv_status:<13} | {rlv_summary}\n"
        f"Environment Configuration{' ':<5}| {env_status:<13} | {env_summary}\n"
        f"RDMA Link Information{' ':<9}| {rdl_status:<13} | {rdl_summary}\n"
        f"NUMA Balancing Information{' ':<4}| {nb_status:<13} | {nb_summary}\n"
        f"{'='*119}\n\n\n"
    )



    # Combine details
    details_width = 120
    details = (
    f"Detailed Output:\n"
    f"{centered_title('OS info', details_width, '=')}\n"
    f"{os_result.stdout.strip()}{os_result.stderr.strip()}\n\n"
    f"{centered_title('ROCm Version', details_width, '=')}\n"
    f"{ROCm_result.stdout.strip()}{ROCm_result.stderr.strip()}\n\n"
    f"{centered_title('HIP Version', details_width, '=')}\n"
    f"{HIP_result.stdout.strip()}{HIP_result.stderr.strip()}\n\n"
    f"{centered_title('Vram Information', details_width, '=')}\n"
    f"{Vram_result.stdout.strip()}{Vram_result.stderr.strip()}\n\n"
    f"{centered_title('UCX Version', details_width, '=')}\n"
    f"{ucx_result.stdout.strip()}{ucx_result.stderr.strip()}\n\n"
    f"{centered_title('MPI Version', details_width, '=')}\n"
    f"{mpi_result.stdout.strip()}{mpi_result.stderr.strip()}\n\n"
    f"{centered_title('Linux Kernel Version', details_width, '=')}\n"
    f"{Lkv_result.stdout.strip()}{Lkv_result.stderr.strip()}\n\n"
    f"{centered_title('Resource limits', details_width, '=')}\n"
    f"{rlv_result.stdout.strip()}{rlv_result.stderr.strip()}\n\n"
    f"{centered_title('Environment Configuration', details_width, '=')}\n"
    f"{env_result.stdout.strip()}{env_result.stderr.strip()}\n\n"
    f"{centered_title('RDMA Link Information', details_width, '=')}\n"
    f"{rdl_result.stdout.strip()}{rdl_result.stderr.strip()}\n\n"
    f"{centered_title('NUMA Balancing Information', details_width, '=')}\n"
    f"{nb_result.stdout.strip()}{nb_result.stderr.strip()}\n\n"
    )
    return summary_table, details


def main():
    hostname = os.uname().nodename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"config.{hostname}.{timestamp}.txt"

    summary_table, details = get_config()


    # Write the summary table and details to the output file
    with open(file_name, "w") as file:
        file.write(summary_table)
        file.write(details)

if __name__ == '__main__':
    main()



# list of stuff to add
# OS version done
# ROCm version done
# GPU VRAM info done
# HIP version done

# PATH
# UCX version done
# MPI version4 done
# MPI version5 done
# ^
# Note from Nilesh applies to 3 above
# these need to change... the /opt/ paths are mostly unique to our setup... other users might have UCX/OMPI at different paths
# the key is that UCX and OMPI should be a part of PATH and LD_LIBRARY_PATH -- first this needs to be checked, and if true, you can simply query ucx_info -v and mpirun --version
# also, we don't need both OMPI4 and OMPI5 check -- usually there's only one of these as part of the env.

# Linux kernel version done
# ulimit -a done
# Environment Variable Config done
# Rdma link info done
# Query Numa balancing status done


# Infiniband device info

# ibstatus
# ibv_devices
# IB_devinfo
# ibstat
# AMDKFD (GPU Driver version) for this one just use DKMS status and put the remainder in the details section


# Network information

# ip a

# ip link

# ip route

# ACSinfo

# rocminfo
# Another note from Nilesh
# rocminfo you need to parse three things -- no. of GPUs, GPU type (gfx___), and Compute Unit count
# -- we can then use this info to parse in the summary one line like "Found 8 MI300X GPUs" or "Found 8 MI308 GPUs"
