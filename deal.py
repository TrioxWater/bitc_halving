"""
This script is used for data preprocessing.
The collected data is converted to a more readable format.
"""
import json
import re
import time
import os

# FILENAME = re.compile(r".+\.json")
# FILENAME = re.compile("./response.json")
# read the file
filelist = ["responce.json"]
# for file in os.listdir():
#     if FILENAME.match(file):
#         filelist.append(file)

# read the json
for file in filelist:

    with open(file, "r") as f:
        print(f"Reading {file}")
        data = json.load(f)
        name = data["name"]
        unit = data["unit"]
        description = data["description"]
        prettified = {
            "name": name,
            "unit": unit,
            "description": description,
        }

        # get the data
        values = data["values"]
        value_pretty = []
        for value in values:
            x = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value["x"]))
            y = value["y"]
            value_pretty.append({
                "x": x,
                "y": y
            })

        prettified["values"] = value_pretty

        # write the data
        with open(file, "w") as f:
            json.dump(prettified, f, indent=4)

            


            


