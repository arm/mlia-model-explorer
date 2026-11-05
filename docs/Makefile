# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

UV ?= uv
CONFIG ?= ../mkdocs.yml
CONTAINER_ENGINE ?= docker
PREVIEW_IMAGE ?= localhost/mlia-docs-preview:0.12.1

help:
	@echo "Targets: build serve preview clean"
	@echo "Run '$(UV) sync --no-install-project --only-group docs' first to install the docs dependencies."

build:
	@$(UV) run --only-group docs mkdocs build --strict -f $(CONFIG)

serve:
	@$(UV) run --only-group docs mkdocs serve -f $(CONFIG)

preview:
	@$(CONTAINER_ENGINE) build -t $(PREVIEW_IMAGE) .
	@$(CONTAINER_ENGINE) run --rm -it -p 127.0.0.1:3000:3000 \
		-v "$(CURDIR):/workspace/docs:ro" $(PREVIEW_IMAGE) --site-id mlia-model-explorer

clean:
	rm -rf ../.mkdocs/site

.PHONY: help build serve preview clean
