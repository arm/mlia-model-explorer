# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

FROM docker.io/library/node:26-slim

ARG NPM_CONFIG_REGISTRY=https://artifactory.arm.com/artifactory/api/npm/mirrors.npmjs_org
ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY} \
    PNPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && echo "deb [trusted=yes] https://artifacts.tools.arm.com/tools-deb bionic main" > /etc/apt/sources.list.d/arm-docs.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends arm-docs-github-action=0.12.1 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm@10.33.3

COPY preview.sh /usr/local/bin/preview-docs
RUN chmod +x /usr/local/bin/preview-docs

EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/preview-docs"]
