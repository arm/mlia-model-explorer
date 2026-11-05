#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -eu

docs_dir=${DOCS_DIR:-/workspace/docs}
if [ ! -f "$docs_dir/source/index.md" ]; then
    printf '%s\n' "Documentation not found: $docs_dir/source/index.md" >&2
    exit 1
fi
if [ ! -f "$docs_dir/docs-config.json" ]; then
    printf '%s\n' "Documentation configuration not found: $docs_dir/docs-config.json" >&2
    exit 1
fi

preview_root=$(mktemp -d)
find "$docs_dir/source" -mindepth 1 -maxdepth 1 \
    -exec ln -s -- {} "$preview_root/" \;
ln -s -- "$docs_dir/docs-config.json" "$preview_root/docs-config.json"

exec arm-docs preview --docs-root "$preview_root" "$@"
