"""Core rocm_docs extension.

It enables a core set of sphinx extensions and provides good defaults for
settings. The environment provided is meant as consistent common base for
ROCm documentation projects.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

import importlib.resources
import inspect
import os
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path

import bs4
import git.repo
from pydata_sphinx_theme.utils import (  # type: ignore[import-untyped]
    config_provided_by_user,
)
from sphinx.application import Sphinx
from sphinx.config import Config

T = TypeVar("T")


class _ConfigUpdater(Generic[T], ABC):
    def __init__(self, default: T) -> None:
        super().__init__()
        self.default = default

    @abstractmethod
    def __call__(self, key: str, app: Sphinx) -> None:
        pass


class _ConfigExtend(_ConfigUpdater[list[T]]):
    def __call__(self, key: str, app: Sphinx) -> None:
        getattr(app.config, key).extend(self.default)


class _ConfigDefault(_ConfigUpdater[T]):
    def __call__(self, key: str, app: Sphinx) -> None:
        if not config_provided_by_user(app, key):
            setattr(app.config, key, self.default)


class _ConfigUnion(_ConfigUpdater[set[T]]):
    def __call__(self, key: str, app: Sphinx) -> None:
        getattr(app.config, key).update(self.default)


class _ConfigMerge(_ConfigUpdater[dict[str, Any]]):
    def __call__(self, key: str, app: Sphinx) -> None:
        current_setting: dict[str, Any] = getattr(app.config, key)
        for item in self.default.items():
            current_setting.setdefault(item[0], item[1])


class _DefaultSettings:
    author = _ConfigDefault(
        'Advanced Micro Devices <a href="https://">Disclaimer and'
        " Licensing Info</a>"
    )
    # pylint: disable=redefined-builtin
    copyright = _ConfigDefault("2022-2023, Advanced Micro Devices Ltd")
    # pylint: enable=redefined-builtin
    myst_enable_extensions = _ConfigUnion(
        {
            "colon_fence",
            "dollarmath",
            "fieldlist",
            "html_image",
            "replacements",
            "substitution",
        }
    )
    myst_heading_anchors = _ConfigDefault(3)
    external_toc_exclude_missing = _ConfigDefault(False)
    epub_show_urls = _ConfigDefault("footnote")
    exclude_patterns = _ConfigExtend(["_build", "Thumbs.db", ".DS_Store"])
    numfig = _ConfigDefault(True)
    linkcheck_timeout = _ConfigDefault(10)
    linkcheck_request_headers = _ConfigMerge(
        {
            r"https://docs.github.com/": {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:112.0)          "
                    "       Gecko/20100101 Firefox/112.0"
                )
            }
        }
    )

    @classmethod
    def update_config(cls, app: Sphinx, _: Config) -> None:
        """Update the Sphinx configuration from the default settings."""
        for name, attr in inspect.getmembers(cls):
            if isinstance(attr, _ConfigUpdater):
                attr(name, app)


def _force_notfound_prefix(app: Sphinx, _: Config) -> None:
    if "READTHEDOCS" not in os.environ:
        return

    if config_provided_by_user(app, "notfound_urls_prefix"):
        return

    components = urllib.parse.urlparse(os.environ["READTHEDOCS_CANONICAL_URL"])
    app.config.notfound_urls_prefix = components.path  # type: ignore[attr-defined]


def _set_article_info(app: Sphinx, _: Config) -> None:
    """Add article info headers to HTML pages."""
    if (
        app.config.setting_all_article_info is False
        and len(app.config.article_pages) == 0
    ):
        return

    article_info = (
        importlib.resources.files("rocm_docs")
        .joinpath("rocm_docs_theme/components/article-info.html")
        .read_text(encoding="utf-8")
    )

    specific_pages: list[str] = []

    _set_page_article_info(app, article_info, specific_pages)

    if app.config.setting_all_article_info is True:
        _set_all_article_info(app, article_info, specific_pages)


def _set_page_article_info(
    app: Sphinx, article_info: str, specific_pages: list[str]
) -> None:
    """Add article info headers to the configured HTML pages.

    The pages can be set in "article_pages" of the Sphinx configuration.
    """
    repo = git.repo.Repo(app.srcdir, search_parent_directories=True)
    for page in app.config.article_pages:
        path_rel = app.project.doc2path(page["file"], basedir=False)
        path_html = Path(app.outdir, path_rel).with_suffix(".html")
        path_source = Path(app.srcdir, path_rel)

        # FIXME: This will silently skip all files when not building the default
        # `html` format (e.g `htmlzip`, `epub` or `pdf`)
        if not path_html.is_file():
            continue

        article_os_info = ""
        if "os" not in page:
            page["os"] = app.config.all_article_info_os
        if "linux" in page["os"]:
            article_os_info += "Linux"
        if "windows" in page["os"]:
            if len(article_os_info) > 0:
                article_os_info += " and "
            article_os_info += "Windows"
        modified_info = article_info.replace("<!--os-info-->", article_os_info)

        author = app.config.all_article_info_author
        if "author" in page:
            author = page["author"]
        modified_info = modified_info.replace("AMD", author)

        date_info: str | None = None
        if "date" in page:
            date_info = page["date"]
        else:
            date_info = _get_time_last_modified(repo, path_source)

        if not date_info:
            date_info = cast(str, app.config.all_article_info_date)

        modified_info = modified_info.replace("2023", date_info)

        if "read-time" in page:
            read_time = page["read-time"]
        else:
            read_time = _estimate_read_time(path_html)
        modified_info = modified_info.replace("5 min read", read_time)

        specific_pages.append(page["file"])
        _write_article_info(path_html, modified_info)


def _set_all_article_info(
    app: Sphinx, article_info: str, specific_pages: list[str]
) -> None:
    """Add article info headers with general settings to all HTML pages.

    Pages that have specific settings (configured by "article_pages") are
    skipped.
    """
    repo = git.repo.Repo(app.srcdir, search_parent_directories=True)
    for docname in app.project.docnames:
        # skip pages with specific settings
        if docname in specific_pages:
            continue

        page_rel = app.project.doc2path(docname, basedir=False)
        page = Path(app.outdir, page_rel).with_suffix(".html")

        # FIXME: This will silently skip all files when not building the default
        # `html` format (e.g `htmlzip`, `epub` or `pdf`)
        if not page.is_file():
            continue

        article_os_info = ""
        if "linux" in app.config.all_article_info_os:
            article_os_info += "Linux"
        if "windows" in app.config.all_article_info_os:
            if len(article_os_info) > 0:
                article_os_info += " and "
            article_os_info += "Windows"

        date_info = _get_time_last_modified(repo, Path(app.srcdir, page_rel))
        if not date_info:
            date_info = cast(str, app.config.all_article_info_date)

        modified_info = article_info.replace("<!--os-info-->", article_os_info)
        modified_info = modified_info.replace(
            "AMD", app.config.all_article_info_author
        )
        modified_info = modified_info.replace("2023", date_info)
        modified_info = modified_info.replace(
            "5 min read", _estimate_read_time(page)
        )

        _write_article_info(page, modified_info)


def _get_time_last_modified(repo: git.repo.Repo, path: Path) -> str | None:
    try:
        time = next(
            repo.iter_commits(paths=path, max_count=1)
        ).committed_datetime
        return time.strftime("%Y-%m-%d")
    except StopIteration:
        return None


def _estimate_read_time(file_name: Path) -> str:
    def is_visible(element):
        if element.parent.name in [
            "style",
            "script",
            "[document]",
            "head",
            "title",
        ]:
            return False
        if isinstance(element, bs4.element.Comment):
            return False
        if element.string == "\n":
            return False
        return True

    words_per_minute = 200
    average_word_length = 5

    with open(file_name, encoding="utf-8") as file:
        html = file.read()
    soup = bs4.BeautifulSoup(html, "html.parser")
    page_text = soup.findAll(text=True)
    visible_page_text = filter(is_visible, page_text)
    average_word_count = (
        sum(len(line) for line in visible_page_text) / average_word_length
    )
    time_minutes = int(max(1, round(average_word_count / words_per_minute)))
    return f"{time_minutes} min read time"


def _write_article_info(path: os.PathLike[Any], article_info: str) -> None:
    with open(path, "r+", encoding="utf8") as file:
        page_html = file.read()
        soup = bs4.BeautifulSoup(page_html, "html.parser")

        has_article_info = soup.find("div", id="rocm-docs-core-article-info")
        if (
            has_article_info is not None
            or soup.article is None
            or soup.article.h1 is None
        ):
            return

        soup.article.h1.insert_after(
            bs4.BeautifulSoup(article_info, "html.parser")
        )
        file.seek(0)
        file.truncate(0)
        file.write(str(soup))


def setup(app: Sphinx) -> dict[str, Any]:
    """Set up rocm_docs.core as a Sphinx extension."""
    required_extensions = [
        "myst_parser",
        "notfound.extension",
        "rocm_docs.projects",
        "sphinx_copybutton",
        "sphinx_design",
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.doctest",
        "sphinx.ext.duration",
    ]
    for ext in required_extensions:
        app.setup_extension(ext)

    app.add_config_value(
        "setting_all_article_info", default=False, rebuild="html", types=str
    )
    app.add_config_value(
        "all_article_info_os",
        default=["linux", "windows"],
        rebuild="html",
        types=str,
    )
    app.add_config_value(
        "all_article_info_author", default="", rebuild="html", types=str
    )
    app.add_config_value(
        "all_article_info_date", default="2023", rebuild="html", types=str
    )
    app.add_config_value(
        "all_article_info_read_time", default="", rebuild="html", types=str
    )
    app.add_config_value("article_pages", default=[], rebuild="html", types=Any)

    # Run before notfound.extension sees the config (default priority(=500))
    app.connect("config-inited", _force_notfound_prefix, priority=400)
    app.connect("config-inited", _DefaultSettings.update_config)
    app.connect("build-finished", _set_article_info, priority=1000)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
