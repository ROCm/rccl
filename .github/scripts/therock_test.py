import json
import os

GITHUB_WORKSPACE = os.getenv("GITHUB_WORKSPACE")

with open('/home/arravikum/cvs/input/mi300_config.json', 'rw') as file:
    data = json.load(file)
    print(data['rocm_path'])
    # point rocm to TheRock
    data['rocm_path'] = f"{GITHUB_WORKSPACE}/build"
    print(data['rocm_path'])
    json.dump(data, file)
