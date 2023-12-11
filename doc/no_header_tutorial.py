from pathlib import Path

PATH = (Path(__file__).parents[1] / "doc" / "tutorial").resolve(strict=True)

for rst in PATH.glob("*.rst"):
    lines = rst.open().readlines()

    for idx, line in enumerate(lines):  # noqa: B007
        if line.startswith("*****"):
            break

    new_lines = [*lines[: idx - 1], *lines[idx + 3 :]]
    rst.open("w").writelines(new_lines)
