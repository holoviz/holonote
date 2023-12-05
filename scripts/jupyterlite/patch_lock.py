import json
import os

path = "pyodide-lock.json"
url = "https://cdn.jsdelivr.net/pyodide/v0.24.1/full"

with open(path) as f:
    data = json.load(f)

for p in data["packages"].values():
    if not p["file_name"].startswith("http"):
        p["file_name"] = os.path.join(url, p["file_name"])

    # Not completely sure why this is empty
    if p["name"] == "holonote":
        p["imports"] = ["holonote"]

with open(path, "w") as f:
    data = json.dump(data, f)
