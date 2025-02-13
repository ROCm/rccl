import subprocess
import time
import os
import re


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


# Get Vram Version
def get_Vram_version():
    result = run_cli_command('rocm-smi --showmeminfo vram')
    if result.stdout:
        summary = "Memory Usage in Vram Information section"
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

    # Vram info
    vram_summary, vram_result = get_Vram_version()
    vram_status = status_check(vram_summary, vram_result)

    


    # Create the summary table
    summary_table = (
        f"\n\n{'='*80}\n"
        f"{'Component':<17}| {'Status':<13} | Value\n"
        f"{'='*80}\n"
        f"OS Version{' ':<7}| {os_status:<13} | {os_summary}\n"
        f"ROCm Version{' ':<5}| {ROCm_status:<13} | {ROCm_summary}\n"
        f"Vram Version{' ':<5}| {vram_status:<13} | {vram_summary}\n"
        f"{'='*80}\n\n\n"
    )



    # Combine details
    details_width = 120
    details = (
    f"Detailed Output:\n"
    f"{centered_title('OS info', details_width, '=')}\n"
    f"{os_result.stdout.strip()}{os_result.stderr.strip()}\n\n"
    f"{centered_title('ROCm Version', details_width, '=')}\n"
    f"{ROCm_result.stdout.strip()}{ROCm_result.stderr.strip()}\n\n"
    f"{centered_title('Vram Information', details_width, '=')}\n\n"
        f"{vram_result.stdout.strip()}{vram_result.stderr.strip()}\n\n"
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
# GPU VRAM info
# HIP version

# UCX version
# MPI version4
# MPI version5
# ^
# Note from Nilesh applies to 3 above
# these need to change... the /opt/ paths are mostly unique to our setup... other users might have UCX/OMPI at different paths
# the key is that UCX and OMPI should be a part of PATH and LD_LIBRARY_PATH -- first this needs to be checked, and if true, you can simply query ucx_info -v and mpirun --version
# also, we don't need both OMPI4 and OMPI5 check -- usually there's only one of these as part of the env.

# Linux kernel version
# ulimit -a
# Environment Variable Config
# Rdma link info
# Query Numa balancing status


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
