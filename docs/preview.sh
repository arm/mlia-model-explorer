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
sync_pid=
server_pid=

sync_docs() {
    rsync -a --delete --exclude=/docs-config.json "$docs_dir/source/" "$preview_root/"
    if ! cmp -s "$docs_dir/docs-config.json" "$preview_root/docs-config.json"; then
        cp -p -- "$docs_dir/docs-config.json" "$preview_root/docs-config.json"
    fi
}

cleanup() {
    for child in "$sync_pid" "$server_pid"; do
        if [ -n "$child" ]; then
            kill "$child" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    rm -rf -- "$preview_root"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sync_docs
(
    while sleep 1; do
        sync_docs
    done
) &
sync_pid=$!

arm-docs preview --docs-root "$preview_root" "$@" &
server_pid=$!
wait "$server_pid"
