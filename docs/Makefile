# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

UV ?= uv
CONFIG ?= ../mkdocs.yml

help:
	@echo "Targets: build serve clean"
	@echo "Run '$(UV) sync --no-install-project --only-group docs' first to install the docs dependencies."

build:
	@$(UV) run --only-group docs mkdocs build --strict -f $(CONFIG)

serve:
	@$(UV) run --only-group docs mkdocs serve -f $(CONFIG)

clean:
	rm -rf ../.mkdocs/site

.PHONY: help build serve clean
