"""Handle external projects.

Remote loading of intersphinx_mapping from file, templating projects in toc.yml).
"""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import functools
import importlib.resources
import json
import os
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import fastjsonschema  # type: ignore[import-untyped]
import github
import requests
import sphinx.util.logging
import yaml
from pydata_sphinx_theme.utils import (  # type: ignore[import-untyped]
    config_provided_by_user,
)
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ExtensionError

from rocm_docs import formatting, util

if sys.version_info < (3, 11):
    import importlib.abc as importlib_abc
else:
    import importlib.resources.abc as importlib_abc

Traversable = importlib_abc.Traversable

Inventory: TypeAlias = str | None | tuple[str | None, ...]
ProjectMapping: TypeAlias = tuple[str, Inventory]

DEFAULT_INTERSPHINX_REPOSITORY = "RadeonOpenCompute/rocm-docs-core"
DEFAULT_INTERSPHINX_BRANCH = "develop"

logger = sphinx.util.logging.getLogger(__name__)


class InvalidMappingFileError(RuntimeError):
    """Mapping file has invalid format, or failed to validate."""


ProjectItem: TypeAlias = str | list[str | None] | None
ProjectEntry: TypeAlias = str | dict[str, ProjectItem]


@dataclass
class _Project:
    target: str
    inventory: list[str | None]
    development_branch: str
    doxygen_html: str | None = None

    @staticmethod
    @functools.lru_cache
    def yaml_schema() -> dict[str, Any]:
        base = importlib.resources.files("rocm_docs") / "data"
        schema_file = base / "projects.schema.json"

        return cast(
            dict[str, Any], json.load(schema_file.open(encoding="utf-8"))
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return cast(dict[str, Any], cls.yaml_schema()["$defs"]["project"])

    @classmethod
    def default_value(cls, prop: str) -> str:
        return cast(str, cls.schema()["properties"][prop]["default"])

    @staticmethod
    def _get_doxygen_html(entry: ProjectEntry) -> str | None:
        assert isinstance(entry, dict)

        if "doxygen" not in entry:
            return None

        doxygen_entry = entry["doxygen"]
        assert isinstance(doxygen_entry, dict | str)

        if isinstance(doxygen_entry, dict):  # type:ignore
            doxygen_entry = doxygen_entry["html"]  # type:ignore

        # Parse as a URI, but only allow the path component
        urlparts = urllib.parse.urlsplit(doxygen_entry)
        for key, value in urlparts._asdict().items():
            if key == "path" or value == "":
                continue
            raise ExtensionError(
                f"URL not allowed for 'doxygen': {doxygen_entry}"
            )
        if urlparts.path.startswith("/"):
            raise ExtensionError(
                f"URL for 'doxygen' must be relative: {doxygen_entry}"
            )

        return urlparts.path

    @classmethod
    def from_yaml_entry(cls, entry: ProjectEntry) -> _Project:
        """Create from an entry that conforms to the project schema."""
        if isinstance(entry, str):
            return _Project(
                entry,
                [cls.default_value("inventory")],
                cls.default_value("development_branch"),
            )

        # It's okay to just index into optional fields, because jsonschema
        # fills in any missing fields with their default values
        inventory = entry["inventory"]
        assert inventory is None or isinstance(inventory, list | str)
        if not isinstance(inventory, list):
            inventory = [inventory]

        return _Project(
            cast(str, entry["target"]),
            inventory,
            cast(str, entry["development_branch"]),
            cls._get_doxygen_html(entry),
        )

    @classmethod
    def get_static_version(
        cls,
        current_branch: str,
        current_project: _Project | None,
    ) -> str | None:
        """Returns a common static version if it exists.

        In some cases all remote projects will receive the same version,
        return that version if this is the case, returns None otherwise.
        """
        # Canonically available everywhere
        if current_branch in ["latest", "stable"]:
            return current_branch

        # Past release versions always with docs/
        if current_branch.startswith("docs-"):
            return current_branch

        # Anything besides the canonical development branch links to latest docs
        development_branch: str = cls.default_value("development_branch")
        if current_project is not None:
            development_branch = current_project.development_branch

        if current_branch != development_branch:
            return "latest"

        return None

    def evaluate(self, static_version: str | None) -> None:
        """Evaluate ${version} placeholders in the inventory and target values."""
        version = (
            static_version
            if static_version is not None
            else self.development_branch
        )
        self.target = self.target.replace("${version}", version)
        for item in self.inventory:
            if item is None:
                continue
            item = item.replace("${version}", version)

    @property
    def mapping(self) -> ProjectMapping:
        """Target and inventory location in the format expected by sphinx."""
        return (self.target, tuple(self.inventory))


def _create_projects(project_yaml: str | Traversable) -> dict[str, _Project]:
    contents = yaml.safe_load(
        project_yaml
        if isinstance(project_yaml, str)
        else project_yaml.open(encoding="utf-8")
    )

    data: dict[str, int | dict[str, ProjectEntry]]
    try:
        data = fastjsonschema.validate(_Project.yaml_schema(), contents)
    except fastjsonschema.exceptions.JsonSchemaValueException as err:
        raise InvalidMappingFileError(
            f"Mapping file is invalid: {err.message}."
        ) from err

    assert isinstance(data["projects"], dict)
    return {
        project_id: _Project.from_yaml_entry(entry)
        for project_id, entry in data["projects"].items()
    }


def _get_current_project(
    projects: dict[str, _Project], current_id: str
) -> _Project | None:
    if current_id in projects:
        return projects[current_id]

    logger.warning(
        f"Current project '{current_id}' not found in projects.\n"
        "Did you forget to set 'external_projects_current_project' to "
        "the name of the current project?"
    )
    return None


def _create_mapping(
    projects: dict[str, _Project],
    current_project: _Project | None,
    current_branch: str,
) -> dict[str, ProjectMapping]:
    static_version = _Project.get_static_version(
        current_branch, current_project
    )
    for project in projects.values():
        project.evaluate(static_version)

    return {name: project.mapping for name, project in projects.items()}


class MappingFileFetchError(RuntimeError):
    """Fetching the yaml file from the remote failed."""


def _fetch_projects(
    remote_repository: str,
    remote_branch: str,
    remote_filepath: str,
) -> str:
    try:
        gh_api = github.Github(os.environ.get("TOKEN"))
        repo = gh_api.get_repo(remote_repository)
        contents = repo.get_contents(remote_filepath, remote_branch)
        if isinstance(contents, list):
            raise MappingFileFetchError("Expected a file not a directory!")

        return contents.decoded_content.decode("utf-8")
    except github.GithubException as err:
        assert isinstance(err.data["message"], str)
        message: str = err.data["message"]
        raise MappingFileFetchError(
            "failed to read remote mappings from "
            f"{remote_repository}:{remote_filepath} "
            f"on branch={remote_branch}, API returned {err.status}: {message}.",
        ) from err


def _load_projects(
    remote_repository: str, remote_branch: str
) -> dict[str, _Project]:
    projects_file_loc = "data/projects.yaml"

    def should_fetch_mappings(
        remote_repository: str | None, remote_branch: str | None
    ) -> bool:
        if not remote_repository:
            logger.info(
                "Skipping the fetch for remote mappings, remote_repository "
                "is unset."
            )
            return False

        if not remote_branch:
            logger.error(
                "Remote branch is unset, cannot fetch remote mappings."
            )
            return False

        logger.info(
            "Remote mappings will be fetched from "
            f"{remote_repository} branch={remote_branch}"
        )
        return True

    projects: dict[str, _Project] | None = None
    if should_fetch_mappings(remote_repository, remote_branch):
        try:
            remote_filepath = "src/rocm_docs/" + projects_file_loc
            projects = _create_projects(
                _fetch_projects(
                    remote_repository,
                    remote_branch,
                    remote_filepath,
                )
            )
        except (MappingFileFetchError, InvalidMappingFileError) as err:
            logger.warning(
                f"Failed to use remote mapping: {err} "
                "Falling back to bundled mapping."
            )

    if projects is None:
        projects = _create_projects(
            importlib.resources.files("rocm_docs") / projects_file_loc
        )

    return projects


def _get_context(
    repo_path: Path, mapping: dict[str, ProjectMapping]
) -> dict[str, Any]:
    url, branch = util.get_branch(repo_path)
    return {
        "url": url,
        "branch": branch,
        "projects": {k: v[0].rstrip("/") for k, v in mapping.items()},
    }


def _update_theme_configs(
    app: Sphinx, current_project: _Project | None, current_branch: str
) -> None:
    """Update configurations for use in theme.py"""
    latest_version = requests.get(
        "https://raw.githubusercontent.com/RadeonOpenCompute/rocm-docs-core/header-versions/latest_version.txt"
    ).text.strip("\r\n")
    latest_version_string = f"docs-{latest_version}"
    release_candidate = requests.get(
        "https://raw.githubusercontent.com/RadeonOpenCompute/rocm-docs-core/header-versions/release_candidate.txt"
    ).text.strip("\r\n")
    release_candidate_string = f"docs-{release_candidate}"

    development_branch = _Project.default_value("development_branch")
    if current_project is not None:
        development_branch = current_project.development_branch

    if current_branch in [latest_version_string, "latest"]:
        app.config.projects_version_type = util.VersionType.LATEST_RELEASE  # type: ignore[attr-defined]
    elif current_branch.startswith(release_candidate_string):
        app.config.projects_version_type = util.VersionType.RELEASE_CANDIDATE  # type: ignore[attr-defined]
    elif current_branch.startswith("docs-"):
        app.config.projects_version_type = util.VersionType.OLD_RELEASE  # type: ignore[attr-defined]
    elif current_branch == development_branch:
        app.config.projects_version_type = util.VersionType.DEVELOPMENT  # type: ignore[attr-defined]


def _get_external_projects(
    app: Sphinx, default: dict[str, ProjectMapping]
) -> list[str]:
    external_projects: list[str] | str = app.config.external_projects
    if external_projects == "all":
        return list(default.keys())
    if isinstance(external_projects, str):
        logger.error(
            f'Unexpected value "{external_projects}" in external_projects.\n'
            'Must be set to a list of project names or "all" to '
            "enable all projects defined in projects.yaml"
        )
        return []

    unknown_projects = [p for p in external_projects if p not in default]
    if len(unknown_projects) > 0:
        known_projects = [f'"{p}"' for p in default]
        unknown_projects = [f'"{p}"' for p in unknown_projects]
        logger.error(
            "Unknown projects: [{}] in external_projects.\n".format(
                ", ".join(unknown_projects)
            )
            + "Valid projects are: [{}]".format(", ".join(known_projects))
        )
    return external_projects


def _set_doxygen_html(app: Sphinx, current_project: _Project | None) -> None:
    if current_project is None or current_project.doxygen_html is None:
        return

    if not hasattr(app.config, "doxygen_html"):
        return

    doxygen_html = current_project.doxygen_html
    if config_provided_by_user(app, "doxygen_html"):
        if doxygen_html != app.config.doxygen_html:
            logger.warning(
                f'The setting doxygen_html="{app.config.doxygen_html}"'
                f' differs from projects.yaml value: "{doxygen_html}"'
            )
        return

    app.config.doxygen_html = doxygen_html  # type: ignore[attr-defined]


def _update_config(app: Sphinx, _: Config) -> None:
    if not config_provided_by_user(app, "intersphinx_disabled_domains"):
        app.config.intersphinx_disabled_domains = ["std"]  # type: ignore[attr-defined]

    remote_repository = app.config.external_projects_remote_repository
    remote_branch = app.config.external_projects_remote_branch
    projects = _load_projects(remote_repository, remote_branch)

    repo_path = Path(app.srcdir)
    __, branch = util.get_branch(repo_path)
    current_project = _get_current_project(
        projects, app.config.external_projects_current_project
    )
    remote_mapping = _create_mapping(projects, current_project, branch)
    external_projects = _get_external_projects(app, remote_mapping)

    mapping: dict[str, ProjectMapping] = app.config.intersphinx_mapping
    for key, value in remote_mapping.items():
        if key in external_projects:
            mapping.setdefault(key, value)

    if not config_provided_by_user(app, "external_toc_path"):
        app.config.external_toc_path = "./.sphinx/_toc.yml"  # type: ignore[attr-defined]

    context = _get_context(Path(app.srcdir), remote_mapping)
    formatting.format_toc(
        Path(app.srcdir, app.config.external_toc_template_path),
        Path(app.srcdir, app.config.external_toc_path),
        context,
    )
    # Store the context to be referenced later
    app.config.projects_context = context  # type: ignore[attr-defined]

    _set_doxygen_html(app, current_project)
    _update_theme_configs(app, current_project, branch)


def _setup_projects_context(
    app: Sphinx, _: str, __: str, context: dict[str, Any], ___: Any
) -> None:
    context["projects"] = app.config.projects_context["projects"]


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup rocm_docs.projects as a sphinx extension."""
    app.setup_extension("sphinx.ext.intersphinx")
    app.setup_extension("sphinx_external_toc")

    app.add_config_value(
        "external_projects_remote_repository",
        DEFAULT_INTERSPHINX_REPOSITORY,
        rebuild="env",
        types=str,
    )
    app.add_config_value(
        "external_projects_remote_branch",
        DEFAULT_INTERSPHINX_BRANCH,
        rebuild="env",
        types=str,
    )
    app.add_config_value(
        "external_projects_current_project",
        lambda config: config.project,
        rebuild="env",
        types=str,
    )
    app.add_config_value(
        "external_projects", "all", rebuild="env", types=[list, str]
    )

    def external_toc_template_default(config: Config) -> str:
        toc_path = Path(config.external_toc_path)
        return str(toc_path.with_suffix(toc_path.suffix + ".in"))

    app.add_config_value(
        "external_toc_template_path",
        external_toc_template_default,
        rebuild="env",
        types=[str, Path],
    )

    # This needs to happen before external-tocs's config-inited (priority=900)
    app.connect("config-inited", _update_config)
    app.connect("html-page-context", _setup_projects_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def debug_projects() -> None:
    """Get remote mappings display them and format the toc.

    Provided as a debugging tool for the functionality of this module.
    """
    projects = _load_projects(
        DEFAULT_INTERSPHINX_REPOSITORY, DEFAULT_INTERSPHINX_BRANCH
    )
    print(projects)

    current_project = _get_current_project(projects, "rocm-docs-core")
    print(current_project)

    repo_path = Path()
    _, branch = util.get_branch(repo_path)
    mapping = _create_mapping(projects, current_project, branch)
    print(mapping)
    context = _get_context(Path(), mapping)
    print(context)
    toc_in = Path("./.sphinx/_toc.yml.in")
    if len(sys.argv) > 1:
        toc_in = Path(sys.argv[1])
    toc_out = toc_in.with_suffix("")
    if len(sys.argv) > 2:
        toc_out = Path(sys.argv[2])

    formatting.format_toc(
        toc_in,
        toc_out,
        context,
    )


if __name__ == "__main__":
    debug_projects()
