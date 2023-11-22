import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest


def main() -> None:
    example_path = Path(__file__).parents[1] / "examples"

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy(example_path / ".." / "pyproject.toml", tmpdir)

        for file in example_path.rglob("*.ipynb"):
            test_folder = tmpdir / uuid4().hex[:8] / file.relative_to(example_path).parent
            test_folder.mkdir(parents=True, exist_ok=True)
            example_file = test_folder / file.name
            shutil.copy(file, example_file)
            shutil.copytree(example_path / "assets", test_folder / "assets")

        os.chdir(tmpdir)
        exit_code = pytest.main(["--nbval-lax", tmpdir])

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
