import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest


def main() -> None:
    example_path = Path(__file__).parents[1] / "examples"

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        os.symlink(example_path / ".." / "pyproject.toml", tmpdir / "pyproject.toml")

        for file in sorted(example_path.rglob("*.ipynb")):
            if file.parent.name == ".ipynb_checkpoints":
                continue
            test_folder = tmpdir / uuid4().hex[:8]
            test_file = test_folder / file.relative_to(example_path)
            test_file.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(file, test_file)
            os.symlink(example_path / "assets", test_folder / "assets")

        os.chdir(tmpdir)
        exit_code = pytest.main(["--nbval-lax", tmpdir])
        os.chdir(example_path)  # To be able to clean up folder on Windows

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
