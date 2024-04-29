"""Module to use rocm-docs-core as a theme."""

from typing import Any

from pathlib import Path

import requests
import sphinx.util.logging
from pydata_sphinx_theme.utils import (  # type: ignore[import-untyped]
    config_provided_by_user,
    get_theme_options_dict,
)
from sphinx.application import Sphinx

from rocm_docs import util

logger = sphinx.util.logging.getLogger(__name__)


def _update_repo_opts(srcdir: str, theme_opts: dict[str, Any]) -> None:
    default_branch_options: dict[str, Any] = {
        "use_edit_page_button": False,
    }
    try:
        url, branch = util.get_branch(srcdir)
        default_branch_options.update(
            {
                "repository_url": url,
                "repository_branch": branch,
                "path_to_docs": util.get_path_to_docs(srcdir),
            }
        )
    except util.InvalidGitRepositoryError:
        logger.warning("Not in a Git Directory, disabling repository buttons")

    for key, val in default_branch_options.items():
        theme_opts.setdefault(key, val)


def _update_banner(
    flavor: str, version_type: util.VersionType, theme_opts: dict[str, Any]
) -> None:
    if flavor != "rocm":
        return

    if version_type == util.VersionType.LATEST_RELEASE:
        return

    announcement_info: str
    if version_type == util.VersionType.RELEASE_CANDIDATE:
        announcement_info = "This page contains changes for a test release of ROCm. Read the <a href='https://rocm.docs.amd.com/en/latest/'>latest Linux release of ROCm documentation</a> for your production environments."
    elif version_type == util.VersionType.OLD_RELEASE:
        announcement_info = "This is an old version of ROCm documentation. Read the <a href='https://rocm.docs.amd.com/en/latest/'>latest ROCm release documentation</a> to stay informed of all our developments."
    elif version_type == util.VersionType.DEVELOPMENT:
        announcement_info = "This page contains proposed changes for a future release of ROCm. Read the <a href='https://rocm.docs.amd.com/en/latest/'>latest Linux release of ROCm documentation</a> for your production environments."

    theme_opts.setdefault("announcement", announcement_info)


def _update_theme_options(app: Sphinx) -> None:
    theme_opts = get_theme_options_dict(app)
    _update_repo_opts(app.srcdir, theme_opts)

    supported_flavors = ["rocm", "local", "rocm-docs-home", "rocm-blogs"]
    flavor = theme_opts.get("flavor", "rocm")
    if flavor not in supported_flavors:
        logger.error(
            f'Unsupported theme flavor "{flavor}", must be one of: {supported_flavors}.\n'
            "Using flavor={supported_flavors[0]}"
        )
        flavor = supported_flavors[0]
        theme_opts["flavor"] = flavor

    theme_opts.setdefault(
        "article_header_start",
        ["components/toggle-primary-sidebar.html", "breadcrumbs.html"],
    )

    if hasattr(app.config, "projects_version_type"):
        _update_banner(flavor, app.config.projects_version_type, theme_opts)

    # Default the download, edit, and fullscreen buttons to off
    for button in ["download", "edit_page", "fullscreen"]:
        theme_opts.setdefault(f"use_{button}_button", False)

    if theme_opts.get("link_main_doc", True):
        theme_opts.setdefault("navbar_center", []).insert(
            0, "components/left-side-menu"
        )

    header_latest_version = requests.get(
        "https://raw.githubusercontent.com/RadeonOpenCompute/rocm-docs-core/header-versions/latest_version.txt"
    ).text.strip("\r\n")
    header_release_candidate_version = requests.get(
        "https://raw.githubusercontent.com/RadeonOpenCompute/rocm-docs-core/header-versions/release_candidate.txt"
    ).text.strip("\r\n")

    default_config_opts = {
        "html_show_sphinx": False,
        "html_favicon": "https://www.amd.com/themes/custom/amd/favicon.ico",
        "notfound_context": {"title": "404 - Page Not Found"},
        "notfound_template": "404.html",
        "html_context": {
            "theme_header_latest_version": header_latest_version,
            "theme_header_release_candidate_version": header_release_candidate_version,
        },
    }
    for key, default in default_config_opts.items():
        if not config_provided_by_user(app, key):
            setattr(app.config, key, default)


def setup(app: Sphinx) -> dict[str, Any]:
    """Set up the module as a Sphinx extension."""
    app.add_js_file(
        "https://download.amd.com/js/analytics/analyticsinit.js",
        priority=999_999,
        loading_method="async",
    )
    app.add_js_file("code_word_breaks.js", loading_method="async")
    app.add_js_file("renameVersionLinks.js", loading_method="async")
    app.add_js_file("rdcMisc.js", loading_method="async")
    app.add_js_file("theme_mode_captions.js", loading_method="async")
    here = Path(__file__).parent.resolve()
    theme_path = here / "rocm_docs_theme"
    app.add_html_theme("rocm_docs_theme", str(theme_path))
    for css in [
        "custom.css",
        "rocm_header.css",
        "rocm_footer.css",
        "fonts.css",
    ]:
        app.add_css_file(css)

    app.connect("builder-inited", _update_theme_options)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
