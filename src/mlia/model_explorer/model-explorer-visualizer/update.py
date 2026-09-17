# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Update the vendored standalone Model Explorer visualizer file."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404: this maintenance script runs the upstream npm build.
from pathlib import Path


def main() -> None:
    """Build and copy the embedded visualizer from a Model Explorer checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_explorer_repo",
        type=Path,
        help="Path to the root of a google-ai-edge/model-explorer checkout.",
    )
    args = parser.parse_args()

    repo = args.model_explorer_repo.expanduser().resolve()
    ui_dir = repo / "src" / "ui"
    if not ui_dir.exists():
        raise SystemExit(f"Model Explorer UI directory not found: {ui_dir}")

    _build_visualizer(ui_dir)
    _copy_visualizer(ui_dir, Path(__file__).resolve().parent)


def _build_visualizer(ui_dir: Path) -> None:
    """Run the upstream custom-element npm package build."""
    subprocess.run(  # nosec B603: npm path is resolved locally and shell is not used.
        [_platform_command("npm"), "run", "build-npm"],
        cwd=ui_dir,
        check=True,
    )


def _platform_command(command: str) -> str:
    if os.name == "nt":
        return f"{command}.cmd"
    return command


def _copy_visualizer(ui_dir: Path, vendored_dir: Path) -> None:
    embedded_bundle = ui_dir / "custom_element_npm" / "dist" / "main_embedded.js"
    _replace_file(embedded_bundle, vendored_dir / "main_embedded.js")


def _replace_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Required build output not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    shutil.copy2(source, destination)


if __name__ == "__main__":
    main()
