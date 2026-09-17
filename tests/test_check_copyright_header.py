# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the copyright header pre-commit hook."""

import datetime
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

import pytest

_ASSERT = unittest.TestCase()


def load_checker_module() -> ModuleType:
    """Load the copyright header checker module from disk."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "pre_commit_hooks"
        / "check_copyright_header.py"
    )
    spec = importlib.util.spec_from_file_location("check_copyright_header", module_path)
    _ASSERT.assertIsNot(spec, None)
    _ASSERT.assertIsNot(spec.loader, None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="checker_module")
def fixture_checker_module() -> ModuleType:
    """Load the copyright header checker module."""
    return load_checker_module()


def test_current_inline_header_passes(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Accept an SPDX copyright header containing the current year."""
    current_year = datetime.datetime.now().year
    source_file = tmp_path / "module.py"
    source_file.write_text(
        f"# SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(source_file)]), True)


def test_missing_header_fails(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a file without an SPDX copyright header."""
    source_file = tmp_path / "module.py"
    source_file.write_text("print('no header')\n", encoding="utf-8")

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(source_file)]), False)
    _ASSERT.assertIn("missing or out of date", capsys.readouterr().out)


def test_outdated_header_fails(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject an SPDX copyright header without the current year."""
    source_file = tmp_path / "module.py"
    source_file.write_text(
        "# SPDX-FileCopyrightText: Copyright 2025, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(source_file)]), False)
    _ASSERT.assertIn("missing or out of date", capsys.readouterr().out)


def test_markdown_header_uses_second_line(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Accept an SPDX header on the second line of a Markdown comment."""
    current_year = datetime.datetime.now().year
    markdown_file = tmp_path / "README.md"
    markdown_file.write_text(
        "<!--\n"
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n"
        "SPDX-License-Identifier: Apache-2.0\n"
        "-->\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(
        checker.check_files_have_updated_header([str(markdown_file)]), True
    )


def test_sidecar_header_takes_precedence(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Accept a current SPDX header from a sidecar file."""
    current_year = datetime.datetime.now().year
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("setting: true\n", encoding="utf-8")
    (tmp_path / "config.yaml.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(yaml_file)]), True)


def test_json_requires_sidecar(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject JSON because it cannot carry an inline header."""
    json_file = tmp_path / "schema.json"
    json_file.write_text('{"schema": true}\n', encoding="utf-8")

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(json_file)]), False)
    _ASSERT.assertIn("requires a sidecar", capsys.readouterr().out)


def test_conflicting_inline_and_sidecar_headers_fail(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject conflicting inline and sidecar copyright owners."""
    current_year = datetime.datetime.now().year
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        f"# SPDX-FileCopyrightText: Copyright {current_year}, Someone Else\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(yaml_file)]), False)
    _ASSERT.assertIn("conflict", capsys.readouterr().out)


def test_matching_inline_and_sidecar_headers_pass(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Accept matching inline and sidecar copyright headers."""
    current_year = datetime.datetime.now().year
    yaml_file = tmp_path / "config.yaml"
    copyright_line = f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited"
    yaml_file.write_text(f"# {copyright_line}\n", encoding="utf-8")
    (tmp_path / "config.yaml.license").write_text(
        f"{copyright_line}\n", encoding="utf-8"
    )

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(checker.check_files_have_updated_header([str(yaml_file)]), True)


def test_missing_files_are_skipped(checker_module: ModuleType) -> None:
    """Ignore deleted files supplied by a diff."""
    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertIs(
        checker.check_files_have_updated_header(["missing-file.txt"]), True
    )


def test_license_text_files_are_skipped(
    tmp_path: Path,
    checker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore canonical license texts containing fixed example copyrights."""
    license_file = tmp_path / "LICENSES" / "Apache-2.0.txt"
    license_file.parent.mkdir()
    license_file.write_text("Copyright [yyyy] [name]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    checker = checker_module.CopyrightHeaderChecker()

    _ASSERT.assertTrue(
        checker.check_files_have_updated_header(["LICENSES/Apache-2.0.txt"])
    )
