import json
import argparse
import os
import csv

def parse(json_file, hip_call, function_name):
    with open(json_file, 'r') as f:
        data = json.load(f)

    kernels=[]
    for entry in data["traceEvents"]:
         if  'name'in entry and entry['name'] == hip_call and function_name in entry['args']['args'] :
            kernels.append(entry)

    sorted_kernels = sorted(kernels, key=lambda x: (x['args']['BeginNs'], x['args']['pid']))

    csv_file_name = os.path.splitext(os.path.basename(json_file))[0] + '_' + function_name + '.csv'
    with open(csv_file_name, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'BeginNs', 'dur', 'ts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        i = 1
        for entry in sorted_kernels:
            record = {'pid': entry['args']['pid'], 'BeginNs': entry['args']['BeginNs'], 'dur': entry['dur'],
                     'ts': entry['ts']}

            writer.writerow(record)
            if (i) % 8 == 0:
                csvfile.write('\n')
                i = 0
    
            i = i + 1
    print(f"Data successfully written to {csv_file_name}")

def main():
    parser = argparse.ArgumentParser(description='Json file and the function to parse.')

    parser.add_argument('json_file', type=argparse.FileType('r'),
                        help='JSON file to load!')
    parser.add_argument('hip_call_name', type=str, help='HIP Call Name, e.g., hipLaunchKernel or hipExtLaunchKernel')
    parser.add_argument('function_name', type=str, help='Kernel Function Name, e.g., gatherTopK, ncclDevKernel_Generic, mscclKernel')

    args = parser.parse_args()
    parse(args.json_file.name, args.hip_call_name, args.function_name) 

if __name__ == '__main__':
    main()

