import subprocess
import time
import os
import re

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


def get_config():
    # Run the commands and store the command outputs

    # OS version
    os_summary, os_result = get_os_version()
    os_status = status_check(os_summary, os_result)


    # Create the summary table
    summary_table = (
        f"\n\n{'='*60}\n"
        f"{'Component':<17}| {'Status':<13} | Value\n"
        f"{'='*60}\n"
        f"OS Version{' ':<7}| {os_status:<13} | {os_summary}\n"
        f"{'='*60}\n\n\n"
    )

    # Combine details
    details = (
        f"{'='*30} OS info {'='*30}\n\n"
        f"{os_result.stdout}{os_result.stderr}\n\n"
        f"{'='*30} Env vars {'='*30}\n\n"
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