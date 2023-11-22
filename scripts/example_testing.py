import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

example_path = Path(__file__).parents[1] / "examples"


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for file in example_path.rglob("*.ipynb"):
            test_folder = tmpdir / str(uuid4()) / file.relative_to(example_path).parent
            test_folder.mkdir(parents=True, exist_ok=True)
            example_file = test_folder / file.name
            shutil.copy(file, example_file)
            shutil.copytree(example_path / "assets", test_folder / "assets")

        pytest.main(["--nbval-lax", tmpdir])


if __name__ == "__main__":
    main()
