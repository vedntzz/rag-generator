"""Enforce CLAUDE.md size limits: functions <= 10 lines, files <= 200 lines."""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

MAX_FUNCTION_LINES = 10
MAX_FILE_LINES = 200
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = ("src", "scripts", "tests")
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def iter_python_files(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        yield from [root] if root.is_file() else sorted(root.rglob("*.py"))


def docstring_line_numbers(node: FunctionNode) -> set[int]:
    if ast.get_docstring(node) is None:
        return set()
    first = node.body[0]
    return set(range(first.lineno, (first.end_lineno or first.lineno) + 1))


def count_function_lines(node: FunctionNode, source_lines: list[str]) -> int:
    # Counts from the def line to the last line, skipping blanks and the docstring.
    skipped = docstring_line_numbers(node)
    span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
    return sum(1 for n in span if n not in skipped and source_lines[n - 1].strip())


def describe_long_function(path: Path, node: FunctionNode, length: int) -> str:
    location = f"{path}:{node.lineno}"
    return f"{location}: function '{node.name}' has {length} lines (max {MAX_FUNCTION_LINES})"


def find_long_functions(path: Path, source: str) -> list[str]:
    lines = source.splitlines()
    functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, FunctionNode)]
    lengths = [(node, count_function_lines(node, lines)) for node in functions]
    too_long = [(node, size) for node, size in lengths if size > MAX_FUNCTION_LINES]
    return [describe_long_function(path, node, size) for node, size in too_long]


def find_long_file(path: Path, source: str) -> list[str]:
    line_count = len(source.splitlines())
    if line_count <= MAX_FILE_LINES:
        return []
    return [f"{path}: file has {line_count} lines (max {MAX_FILE_LINES})"]


def find_violations(roots: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in iter_python_files(roots):
        source = path.read_text(encoding="utf-8")
        violations += find_long_file(path, source) + find_long_functions(path, source)
    return violations


def main(argv: list[str]) -> int:
    default_roots = [REPO_ROOT / name for name in DEFAULT_ROOTS if (REPO_ROOT / name).exists()]
    violations = find_violations([Path(arg) for arg in argv] or default_roots)
    for violation in violations:
        print(violation)
    print(f"check_limits: {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
