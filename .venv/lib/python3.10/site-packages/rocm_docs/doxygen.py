"""Doxygen sub-extension of rocm-docs-core."""

from __future__ import annotations

from typing import Any, Union, cast

import importlib.metadata
import importlib.resources
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydata_sphinx_theme.utils import (  # type: ignore[import-untyped]
    config_provided_by_user,
)
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ConfigError, ExtensionError
from sphinx.util import logging, progress_message
from sphinx.util.osutil import copyfile

from rocm_docs import util

logger = logging.getLogger(__name__)


def _copy_files(app: Sphinx) -> None:
    """Insert additional files into workspace."""
    pkg = importlib.resources.files("rocm_docs")
    Path(app.srcdir, "_doxygen").mkdir(exist_ok=True)
    util.copy_from_package(
        app, pkg / "data/_doxygen", "data/_doxygen", "_doxygen"
    )


def _get_config_default(config: Config, key: str) -> Any:
    """Get the default value for a sphinx configuration key."""
    default_or_callable = config.values[key][0]
    if callable(default_or_callable):
        return default_or_callable(config)
    return default_or_callable


def _run_doxygen(app: Sphinx, config: Config) -> None:
    """Run doxygen, validating all paths."""
    # To support the (legacy) ROCmDocs interface 'None' is a synonym for the
    # default value
    for i in ["doxygen_root", "doxyfile"]:
        if getattr(config, i) is None:
            setattr(config, i, _get_config_default(config, i))

    doxygen_root = Path(app.confdir, config.doxygen_root)
    try:
        doxygen_root = doxygen_root.resolve(strict=True)
    except FileNotFoundError as err:
        raise ConfigError(
            f"Expected doxygen root folder {doxygen_root} to exist"
            " and be readable."
        ) from err

    if config["doxygen_executable"] is None:
        cmd_path = shutil.which("doxygen")
        if cmd_path is None:
            raise RuntimeError(
                "'doxygen' command not found! Make sure that "
                "doxygen is installed and in the PATH or specify "
                "via doxygen_executable configuration variable."
            )
        doxygen_exe = Path(cmd_path)
    else:
        doxygen_exe = Path(app.confdir, config.doxygen_executable)

    doxyfile = Path(app.confdir, config["doxyfile"]).absolute()
    if not doxyfile.is_file():
        raise ConfigError(
            f"Expected doxyfile {doxyfile} to exist and be readable."
        )

    # Running doxygen requires that the files are already copied because
    # the Doxyfile references files distributed with rocm-docs-core
    # (e.g. stylesheets)
    if app.config.doxysphinx_enabled:
        _copy_files(app)
    try:
        subprocess.check_call([doxygen_exe, "--version"], cwd=doxygen_root)
        subprocess.check_call([doxygen_exe, doxyfile], cwd=doxygen_root)
    except subprocess.CalledProcessError as err:
        raise RuntimeError("Failed when running doxygen") from err

    _update_breathe_settings(app, doxygen_root)
    _run_doxysphinx(app, doxygen_root, doxyfile, doxygen_exe)


def _update_breathe_settings(app: Sphinx, doxygen_root: Path) -> None:
    if config_provided_by_user(
        app, "breathe_projects"
    ) or config_provided_by_user(app, "breathe_default_project"):
        return

    doxygen_project: dict[str, None | str | os.PathLike[Any]] = (
        app.config.doxygen_project
    )

    # To support the (legacy) ROCmDocs interface 'None' is a synonym for the
    # default value for each element of the Tuple
    default: dict[str, None | str | os.PathLike[Any]] = _get_config_default(
        app.config, "doxygen_project"
    )
    for key, value in doxygen_project.items():
        if value is None:
            doxygen_project[key] = default[key]

    # FIXME: Log an error and reset to default instead of asserting
    assert isinstance(doxygen_project["name"], str)
    project_name: str = doxygen_project["name"]

    # First try relative to the
    assert doxygen_project["path"] is not None
    xml_path = Path(app.confdir, doxygen_project["path"])
    try:
        xml_path = xml_path.resolve(strict=True)
    except FileNotFoundError as err:
        raise NotADirectoryError(
            "Expected doxygen to generate the folder"
            f" {doxygen_root!s} but it could not be found."
        ) from err

    setattr(app.config, "breathe_projects", {project_name: str(xml_path)})
    setattr(app.config, "breathe_default_project", project_name)


def _update_doxylink_settings(app: Sphinx, _: Config) -> None:
    if not hasattr(app.config, "doxylink"):
        logger.info(
            "doxylink not enabled, skipping setting up the current" " project"
        )
        return

    if app.config.doxygen_html is None:
        return

    # Materialize the default value, since we're about to mutate it
    # Otherwise config.doxylink would return a temporary and any modification to
    # it would be lost.
    if not config_provided_by_user(app, "doxylink"):
        app.config.doxylink = app.config.doxylink  # type: ignore [attr-defined]

    doxylink = cast(
        dict[str, tuple[str, str] | tuple[str, str, str]],
        app.config.doxylink,
    )
    tagfile = Path(app.srcdir, app.config.doxygen_html, "tagfile.xml")

    doxylink.setdefault("doxygen", (str(tagfile), str(app.config.doxygen_html)))


def _run_doxysphinx(
    app: Sphinx, doxygen_root: Path, doxyfile: Path, doxygen_exe: Path
) -> None:
    if not app.config.doxysphinx_enabled:
        return

    if (
        "doxysphinx.cli" not in sys.modules
        and importlib.util.find_spec("doxysphinx.cli") is None
    ):
        raise RuntimeError(
            "Missing optional dependencies: make sure the "
            "[api_reference] feature is enabled. (e.g. "
            '"pip install rocm-docs-core[api_reference]")'
        )

    doxyphinx_version = importlib.metadata.version("doxysphinx")
    args = [
        sys.executable,
        "-m",
        "doxysphinx",
        "build",
        "--doxygen_exe=" + str(doxygen_exe.absolute()),
    ]
    if doxyphinx_version.endswith("+tagfile.toc"):
        args.append("--tagfile_toc")
    args += [app.srcdir, app.outdir, str(doxyfile)]

    try:
        subprocess.check_call(args, cwd=doxygen_root)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"doxysphinx failed (exit code: {err.returncode})"
        ) from err


def _copy_tagfile(app: Sphinx) -> None:
    if app.config.doxygen_html is None:
        return

    if app.builder.format != "html":
        return

    src = Path(app.srcdir, app.config.doxygen_html, "tagfile.xml")
    dst = Path(app.outdir, src.name)
    with progress_message("copying doxygen tagfile"):
        try:
            copyfile(str(src), str(dst))
        except OSError as err:
            raise ExtensionError(f"Failed to copy tag file: {err}") from err


def setup(app: Sphinx) -> dict[str, Any]:
    """Set up rocm_docs.doxygen as a Sphinx extension."""
    app.setup_extension("sphinx.ext.mathjax")
    app.setup_extension("breathe")

    app.add_config_value(
        "doxygen_root", ".doxygen", rebuild="", types=[None, str, os.PathLike]
    )
    app.add_config_value(
        "doxygen_executable",
        None,
        rebuild="",
        types=[None, str, "os.PathLike[Any]"],
    )
    app.add_config_value(
        "doxyfile",
        lambda config: Path(config.doxygen_root, "Doxyfile"),
        rebuild="",
        types=[None, str, "os.PathLike[Any]"],
    )
    app.add_config_value(
        "doxygen_project",
        lambda config: {
            "name": "doxygen",
            "path": Path(config.doxygen_root, "docBin", "xml"),
        },
        rebuild="",
        types=dict[str, Union[None, str, "os.PathLike[Any]"]],
    )
    app.add_config_value("doxysphinx_enabled", False, rebuild="", types=bool)
    app.add_config_value("doxygen_html", None, rebuild="")

    # Should run before breathe sees their parameters, as we provide defaults.
    app.connect("config-inited", _run_doxygen, priority=400)
    # Has to run after projects.py config-inited, as it might set
    # doxygen_html
    app.connect("config-inited", _update_doxylink_settings, priority=500)
    # Should run after projects.py's config (if enabled) as it provides values
    # based on the contents projects.yaml, needs access to the builder
    app.connect("builder-inited", _copy_tagfile)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
