# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual vendored visualizer update helper."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


def _load_updater() -> ModuleType:
    script = (
        Path(__file__).parents[1]
        / "src/mlia/model_explorer/model-explorer-visualizer/update.py"
    )
    spec = importlib.util.spec_from_file_location("visualizer_update", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def update_case(tmp_path: Path) -> tuple[ModuleType, Path, Path, bytes, bytes]:
    """Create a matched new pair and an existing vendored pair."""
    ui_dir = tmp_path / "ui"
    source = ui_dir / "custom_element_npm/dist"
    source.mkdir(parents=True)
    bundle = b"new visualizer bundle\n"
    digest = hashlib.sha256(bundle).hexdigest()
    license_bytes = f"Combined notices\nBundle-SHA256: {digest}\n".encode()
    (source / "main_embedded.js").write_bytes(bundle)
    (source / "main_embedded.js.license").write_bytes(license_bytes)
    vendored = tmp_path / "vendored"
    vendored.mkdir()
    (vendored / "main_embedded.js").write_bytes(b"old bundle")
    (vendored / "main_embedded.js.license").write_bytes(b"old notices")
    return _load_updater(), ui_dir, vendored, bundle, license_bytes


def test_update_copies_bundle_and_matching_notices(update_case: tuple) -> None:
    updater, ui_dir, vendored, bundle, notices = update_case
    updater._copy_visualizer(ui_dir, vendored)
    assert (vendored / "main_embedded.js").read_bytes() == bundle
    assert (vendored / "main_embedded.js.license").read_bytes() == notices
    assert sorted(p.name for p in vendored.iterdir()) == [
        "main_embedded.js", "main_embedded.js.license"
    ]


@pytest.mark.parametrize("missing", ["main_embedded.js", "main_embedded.js.license"])
def test_missing_output_preserves_both_vendored_files(
    update_case: tuple, missing: str
) -> None:
    updater, ui_dir, vendored, _, _ = update_case
    (ui_dir / "custom_element_npm/dist" / missing).unlink()
    with pytest.raises(SystemExit, match="Required build output not found"):
        updater._copy_visualizer(ui_dir, vendored)
    assert (vendored / "main_embedded.js").read_bytes() == b"old bundle"
    assert (vendored / "main_embedded.js.license").read_bytes() == b"old notices"


@pytest.mark.parametrize("notices", [
    b"unbound notices\n",
    b"Bundle-SHA256: " + b"0" * 64 + b"\n",
    b"",
])
def test_mismatched_or_unbound_notices_preserve_both_files(
    update_case: tuple, notices: bytes
) -> None:
    updater, ui_dir, vendored, _, _ = update_case
    (ui_dir / "custom_element_npm/dist/main_embedded.js.license").write_bytes(notices)
    with pytest.raises(SystemExit, match="license does not match"):
        updater._copy_visualizer(ui_dir, vendored)
    assert (vendored / "main_embedded.js").read_bytes() == b"old bundle"
    assert (vendored / "main_embedded.js.license").read_bytes() == b"old notices"


def test_staging_failure_does_not_replace_either_file(update_case: tuple) -> None:
    updater, ui_dir, vendored, _, _ = update_case
    original_copy = updater.shutil.copy2

    def fail_license_copy(source: Path, destination: Path) -> None:
        if source.name.endswith(".license"):
            raise OSError("simulated copy failure")
        original_copy(source, destination)

    with patch.object(updater.shutil, "copy2", side_effect=fail_license_copy):
        with pytest.raises(OSError, match="simulated copy failure"):
            updater._copy_visualizer(ui_dir, vendored)
    assert (vendored / "main_embedded.js").read_bytes() == b"old bundle"
    assert (vendored / "main_embedded.js.license").read_bytes() == b"old notices"
    assert not list(vendored.glob(".visualizer-update-*"))


def test_staged_bytes_are_validated_before_replacement(update_case: tuple) -> None:
    updater, ui_dir, vendored, _, _ = update_case
    original_copy = updater.shutil.copy2

    def copy_changed_bundle(source: Path, destination: Path) -> None:
        original_copy(source, destination)
        if source.name == "main_embedded.js":
            destination.write_bytes(b"changed during copying")

    with patch.object(updater.shutil, "copy2", side_effect=copy_changed_bundle):
        with pytest.raises(SystemExit, match="license does not match"):
            updater._copy_visualizer(ui_dir, vendored)
    assert (vendored / "main_embedded.js").read_bytes() == b"old bundle"
    assert (vendored / "main_embedded.js.license").read_bytes() == b"old notices"


def test_main_builds_then_copies_generated_outputs(update_case: tuple) -> None:
    updater, ui_dir, vendored, bundle, notices = update_case
    repo = ui_dir.parents[0]
    real_ui = repo / "src/ui"
    real_ui.parent.mkdir()
    ui_dir.rename(real_ui)
    updater.__file__ = str(vendored / "update.py")
    with patch("sys.argv", ["update.py", str(repo)]):
        with patch.object(updater, "_build_visualizer") as build:
            updater.main()
    build.assert_called_once_with(real_ui)
    assert (vendored / "main_embedded.js").read_bytes() == bundle
    assert (vendored / "main_embedded.js.license").read_bytes() == notices
