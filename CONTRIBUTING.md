<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Contribution Guidelines

ML Inference Advisor (MLIA) welcomes external contributions. Contributions
must be licensed under [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)
and meet the checks below before they can be merged.

## Setting up the plugin repository

Clone the repository and install its development dependencies:

```bash
git clone https://github.com/arm/mlia-model-explorer.git
cd mlia-model-explorer
uv sync --group dev
```

Set your Git name to your real name and use the email address associated with
your GitHub account. Pseudonymous or anonymous contributions are not accepted.
Shared MLIA APIs live in the [`arm/mlia` repository](https://github.com/arm/mlia).

## Checks

Run the tests and pre-commit checks before opening a pull request:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Pre-commit checks may reformat files. You can install the hooks with
`uv run pre-commit install`. For local sibling checkouts, browser visualizer
updates, and documentation builds, follow the
[development guide](docs/source/development.md).

## Commits and review

Use [Conventional Commits](https://www.conventionalcommits.org/) with a
capitalized header description. Allowed types are `build`, `ci`, `docs`,
`feat`, `fix`, `perf`, `refactor`, `style`, and `test`.

Sign off every commit to certify its origin under the
[Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/):

```bash
git commit -s -m "fix: Describe the change"
```

The sign-off adds `Signed-off-by: FIRST_NAME SECOND_NAME <your@email.address>`
to the commit message. Contributions are reviewed through GitHub pull requests
and must pass the repository's CI checks before merging.

## Reporting bugs

Use the [`arm/mlia-model-explorer` issue tracker](https://github.com/arm/mlia-model-explorer/issues)
for plugin-specific bugs. For shared MLIA behavior, use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues).

## Copyright notices

Include the appropriate SPDX copyright and Apache-2.0 license notice in each
source file. Update an existing file's notice when your changes require it.
