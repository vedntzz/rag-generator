"""Tests for scripts/check_limits.py and enforcement of the limits on this repo."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_LIMITS_SCRIPT = REPO_ROOT / "scripts" / "check_limits.py"


def run_check_limits(*targets: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(CHECK_LIMITS_SCRIPT), *map(str, targets)]
    return subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)


def write_source(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source)
    return path


def function_source(total_lines: int, prefix: str = "def") -> str:
    # One signature line followed by (total_lines - 1) body statements.
    return f"{prefix} sample_function() -> None:\n" + "    x = 0\n" * (total_lines - 1)


@pytest.fixture
def eleven_line_function_file(tmp_path: Path) -> Path:
    return write_source(tmp_path, function_source(11))


@pytest.fixture
def ten_line_function_file(tmp_path: Path) -> Path:
    return write_source(tmp_path, function_source(10))


@pytest.fixture
def eleven_line_async_function_file(tmp_path: Path) -> Path:
    return write_source(tmp_path, function_source(11, prefix="async def"))


@pytest.fixture
def ten_code_lines_with_docstring_and_blanks_file(tmp_path: Path) -> Path:
    docstring = '    """Docstring line one.\n\n    Docstring line three.\n    """\n'
    body = "    x = 0\n\n" * 9
    return write_source(tmp_path, "def sample_function() -> None:\n" + docstring + body)


@pytest.fixture
def two_hundred_one_line_file(tmp_path: Path) -> Path:
    return write_source(tmp_path, "x = 0\n" * 201)


@pytest.fixture
def two_hundred_line_file(tmp_path: Path) -> Path:
    return write_source(tmp_path, "x = 0\n" * 200)


def test_check_limits_flags_function_over_ten_lines(eleven_line_function_file: Path) -> None:
    result = run_check_limits(eleven_line_function_file)
    assert result.returncode == 1
    assert "sample_function" in result.stdout


def test_check_limits_passes_function_of_exactly_ten_lines(ten_line_function_file: Path) -> None:
    result = run_check_limits(ten_line_function_file)
    assert result.returncode == 0, result.stdout


def test_check_limits_flags_async_function_over_ten_lines(
    eleven_line_async_function_file: Path,
) -> None:
    result = run_check_limits(eleven_line_async_function_file)
    assert result.returncode == 1
    assert "sample_function" in result.stdout


def test_check_limits_ignores_docstring_and_blank_lines_in_function_length(
    ten_code_lines_with_docstring_and_blanks_file: Path,
) -> None:
    result = run_check_limits(ten_code_lines_with_docstring_and_blanks_file)
    assert result.returncode == 0, result.stdout


def test_check_limits_flags_file_over_two_hundred_lines(two_hundred_one_line_file: Path) -> None:
    result = run_check_limits(two_hundred_one_line_file)
    assert result.returncode == 1
    assert "201" in result.stdout


def test_check_limits_passes_file_of_exactly_two_hundred_lines(two_hundred_line_file: Path) -> None:
    result = run_check_limits(two_hundred_line_file)
    assert result.returncode == 0, result.stdout


def test_check_limits_scans_directories_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text(function_source(11))
    result = run_check_limits(tmp_path)
    assert result.returncode == 1


def test_repository_respects_code_limits() -> None:
    result = run_check_limits()
    assert result.returncode == 0, result.stdout
