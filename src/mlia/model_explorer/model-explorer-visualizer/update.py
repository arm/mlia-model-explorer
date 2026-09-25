# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Update the vendored standalone Model Explorer visualizer and its notices."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess  # nosec B404: this maintenance script runs the upstream npm build.
import tempfile
from pathlib import Path

_BUNDLE_NAME = "main_embedded.js"
_LICENSE_NAME = f"{_BUNDLE_NAME}.license"


def main() -> None:
    """Build and copy the embedded visualizer and matching license file."""
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
    """Validate and stage both generated files before replacing either one."""
    source_dir = ui_dir / "custom_element_npm" / "dist"
    names = (_BUNDLE_NAME, _LICENSE_NAME)
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise SystemExit(f"Required build output not found: {source}")

    vendored_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".visualizer-update-", dir=vendored_dir
    ) as tmp:
        staging_dir = Path(tmp)
        for name in names:
            shutil.copy2(source_dir / name, staging_dir / name)
        # Validate the staged bytes, even if the build output changed during copying.
        license_text = (staging_dir / _LICENSE_NAME).read_text(encoding="utf-8")
        hashes = re.findall(
            r"^Bundle-SHA256: ([0-9a-f]{64})$", license_text, re.MULTILINE
        )
        bundle_hash = hashlib.sha256(
            (staging_dir / _BUNDLE_NAME).read_bytes()
        ).hexdigest()
        if hashes != [bundle_hash]:
            raise SystemExit("Generated license does not match main_embedded.js")
        for name in names:
            # Replace the directory entry, not the target of an existing symlink.
            os.replace(staging_dir / name, vendored_dir / name)


if __name__ == "__main__":
    main()
