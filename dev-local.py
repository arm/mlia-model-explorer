# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    overrides_file = project_dir / "local-overrides.txt"

    if not overrides_file.exists():
        print(f'Missing local override file: "{overrides_file}"', file=sys.stderr)
        print(
            "Create it from the example in docs/source/development.md.",
            file=sys.stderr,
        )
        return 1

    command = [
        "uv",
        "run",
        "--project",
        str(project_dir),
        "--with-requirements",
        str(overrides_file),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
