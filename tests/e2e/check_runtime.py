# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Export real models through an installed runtime and check frontend isolation.

Run with: uv run python tests/e2e/check_runtime.py MODEL [MODEL ...]
"""

import argparse
import json
import sys
from importlib.metadata import distributions
from pathlib import Path

from mlia.core.context import ExecutionContext
from mlia.plugins.analysis import AnalysisRunResult

from mlia.model_explorer.artifacts import write_artifacts
from mlia.model_explorer.bundler import build_artifacts_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--execution-profile")
    parser.add_argument("--output-dir", type=Path, default=Path("report/runtime-e2e"))
    args = parser.parse_args()
    before = {d.metadata["Name"]: d.version for d in distributions()}
    for index, model in enumerate(args.models):
        output = (args.output_dir / str(index)).resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = AnalysisRunResult(
            output=None,
            context=ExecutionContext(output_dir=output),
            args={},
            command_name="check",
            parameters={"model": model.resolve(), "noninteractive": True},
        )
        artifacts = write_artifacts(result, execution_profile=args.execution_profile)
        manifest = json.loads(artifacts.manifest.read_text())
        graph = json.loads((artifacts.root / manifest["model_path"]).read_text())
        assert graph["graphs"] and any(g["nodes"] for g in graph["graphs"])
        assert build_artifacts_bundle(artifacts.root).is_file()
        assert before == {d.metadata["Name"]: d.version for d in distributions()}
        assert (
            not {"torch", "executorch", "onnx", "model_explorer"} & sys.modules.keys()
        )
        print(f"PASS: {model.name}; bundle generated, frontend unchanged", flush=True)


if __name__ == "__main__":
    main()
