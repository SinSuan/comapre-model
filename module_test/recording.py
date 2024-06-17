import os
import json

def save_record(model_record, path_folder, name_file):
    print("enter save_record")

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    path_file = f"{path_folder}/{name_file}"
    if os.path.exists(f"{path_file}.json"):
        idx = 2
        path_temp = f"{path_file}_{idx}"
        while(os.path.exists(f"{path_temp}.json")):
            idx += 1
            path_temp = f"{path_file}_{idx}"
        path_file = path_temp

    with open(f"{path_file}.json", "w") as f:
        json_object = json.dumps(model_record, ensure_ascii=False, indent=4)
        f.write(json_object)
        f.close()

    print("exit save_record")
    return f"{path_file}.json"

def load_record(path_folder, name_file):
    print("enter load_record")
    path_file = f"{path_folder}/{name_file}"
    with open(f"{path_file}.json", "r") as f:
        model_record = json.load(f)
    result = model_record["result"]
    print("exit load_record")
    return result
