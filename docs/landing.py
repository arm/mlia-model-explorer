# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Render the shared landing content with MkDocs Material."""

import json
import re
from pathlib import Path
from shutil import copyfile
from typing import Any
from urllib.parse import urlsplit

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.nav import Navigation
from mkdocs.structure.pages import Page
from mkdocs.utils import get_relative_url


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Keep the existing Markdown homepage available as documentation."""
    source = Path(config.docs_dir) / "index.md"
    files.append(File.generated(config, "documentation.md", content=source.read_text()))
    return files


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    """Choose the landing template only for the homepage."""
    if page.file.src_uri == "index.md":
        page.meta["template"] = "landing.html"
        page.meta["hide"] = ["navigation", "toc"]
        return ""
    return markdown


def on_page_context(
    context: dict[str, Any], page: Page, config: MkDocsConfig, nav: Navigation
) -> dict[str, Any]:
    """Expose shared content and links that also work in offline builds."""
    if page.file.src_uri != "index.md":
        return context
    source = Path(config.docs_dir) / "landing-page"
    context["landing"] = json.loads((source / "content.json").read_text())
    context["is_core"] = "capabilities" in context["landing"]

    def landing_url(url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            return url
        destination = parts.path.strip("/")
        if destination == "documentation":
            target = next(
                item.file
                for item in nav.pages
                if item.file.src_uri == "documentation.md"
            )
        elif destination:
            target = next(
                item.file
                for item in nav.pages
                if item.file.src_uri == destination + ".md"
            )
        else:
            target = page.file
        relative = str(get_relative_url(target.url, page.file.url))
        return relative + ("#" + parts.fragment if parts.fragment else "")

    context["landing_url"] = landing_url
    return context


def on_post_build(config: MkDocsConfig) -> None:
    """Publish logos and a scoped version of the shared landing stylesheet."""
    source = Path(config.docs_dir) / "landing-page"
    output = Path(config.site_dir) / "assets" / "mlia"
    output.mkdir(parents=True, exist_ok=True)
    for image in source.glob("mlia-logo-stacked-*.png"):
        copyfile(image, output / image.name)
    css = (source / "index.module.css").read_text()
    css = css.replace(
        ':global(html[data-theme="dark"])', '[data-md-color-scheme="slate"]'
    )
    css = re.sub(r"(?<![\w-])\.([a-zA-Z_][\w-]*)", r".mlia-\1", css)
    css += """
.mlia-home { margin: 0 -0.8rem; }
.mlia-home .mlia-brand img { width: 100%; height: auto; }
.mlia-logo-dark { display: none !important; }
[data-md-color-scheme="slate"] .mlia-logo-light { display: none !important; }
[data-md-color-scheme="slate"] .mlia-logo-dark { display: block !important; }
.mlia-home h1, .mlia-home h2, .mlia-home h3 { font-weight: 500; }
.mlia-home .mlia-hero h1 { color: var(--mlia-ink); font-weight: 600; }
.mlia-home .mlia-analysisConsole pre {
 background: #f3f3f3; color: #080225; padding: 1.25rem;
 border-radius: 0.5rem; overflow: auto;
}
[data-md-color-scheme="slate"] .mlia-home .mlia-analysisConsole pre {
 background: #080225; color: #fff;
}
"""
    css += """
.mlia-home a.mlia-primaryButton,
.mlia-home a.mlia-primaryButton:visited { color: #fff; }
[data-md-color-scheme="slate"] .mlia-home a.mlia-primaryButton { color: #080225; }
.mlia-home a.mlia-secondaryButton,
.mlia-home a.mlia-capabilityCard,
.mlia-home a.mlia-pluginCard,
.mlia-home a.mlia-guideRow { color: var(--mlia-ink); }
.mlia-home a.mlia-repositoryLink,
.mlia-home a.mlia-cardLink,
.mlia-home .mlia-textLinks a { color: var(--mlia-accent); }
"""
    (output / "landing.css").write_text(css)
