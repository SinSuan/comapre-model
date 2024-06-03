import os

def save_record(model_record, path_file):
    print("enter save_record")

    if os.path.exists(f"{path_file}.json"):
        idx = 2
        path_temp = f"{path_file}_{idx}.json"
        while(os.path.exists(path_temp)):
            idx += 1
            path_temp = f"{path_file}_{idx}.json"
        path_file = path_temp

    with open(f"path_file.json", "w") as f:
        f.write(model_record)

    print("exit save_record")
