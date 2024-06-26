import hashlib
import json
import os
from glob import glob


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


with open("package.json") as f:
    package = json.load(f)
pyodide_version = package["dependencies"]["pyodide"].removeprefix("^")

path = "pyodide-lock.json"
url = f"https://cdn.jsdelivr.net/pyodide/v{pyodide_version}/full"

with open(path) as f:
    data = json.load(f)

for p in data["packages"].values():
    if not p["file_name"].startswith("http"):
        p["file_name"] = os.path.join(url, p["file_name"])


# Special handling of holonote
whl_file = glob("../../dist/*.whl")[0]
hn = data["packages"]["holonote"]
hn["version"] = os.environ["VERSION"]
hn["file_name"] = os.path.basename(whl_file)
hn["sha256"] = calculate_sha256(whl_file)
hn["imports"] = ["holonote"]  # Not completely sure why this is empty

# To avoid importing it in the notebooks, we can't add it to pandas directly
# as fastparquet depends on it. So we add it to hvplot instead.
data["packages"]["hvplot"]["depends"].extend(["fastparquet"])

data["packages"]["holoviews"]["depends"].extend(["pyparsing"])


with open(path, "w") as f:
    data = json.dump(data, f)
