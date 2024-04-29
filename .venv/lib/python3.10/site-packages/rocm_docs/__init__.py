"""Defines the ROCmDocs package.

Set up variables for documentation of ROCm projects
that are using Read the Docs.
"""

from __future__ import annotations

from typing import ClassVar, TypeAlias

import os

from rocm_docs.core import setup

MaybePath: TypeAlias = str | os.PathLike[str] | None


# Intentionally disabling the too-many-instance-attributes check in pylint
# as this class is intended to contain all necessary Sphinx config variables
# pylint: disable=too-many-instance-attributes
class ROCmDocs:
    """A class to contain all of the Sphinx variables."""

    SPHINX_VARS: ClassVar = [
        "extensions",
        "html_title",
        "html_theme",
        "html_theme_options",
        "doxygen_root",
        "doxygen_project",
        "doxyfile",
        "doxysphinx_enabled",
    ]

    def __init__(
        self,
        project_name: str,
        _: str | None = None,
        __: MaybePath = None,
    ) -> None:
        """Intialize ROCmDocs."""
        self._project_name: str = project_name
        self.extensions: list[str] = []
        self.html_title: str
        self.html_theme: str
        self.html_theme_options: dict[str, str | (bool | list[str])] = {}
        self.doxygen_root: MaybePath = None
        self.doxygen_project: dict[str, str | None | MaybePath] = {
            "name": None,
            "path": None,
        }
        self.doxyfile: MaybePath = None
        self.doxysphinx_enabled: bool = False

    @property
    def project(self) -> str:
        """Sphinx project variable."""
        return self._project_name

    def run_doxygen(
        self,
        doxygen_root: MaybePath = None,
        doxygen_path: MaybePath = None,
        doxygen_file: str | None = None,
    ) -> None:
        """Run doxygen as part of Sphinx by adding rocm_docs.doxygen."""
        if "rocm_docs.doxygen" not in self.extensions:
            self.extensions.append("rocm_docs.doxygen")

        self.doxygen_root = doxygen_root
        self.doxygen_project = {
            "name": self._project_name,
            "path": doxygen_path,
        }
        self.doxyfile = doxygen_file

    def enable_api_reference(self) -> None:
        """Enable embedding the doxygen generated api."""
        if "rocm_docs.doxygen" not in self.extensions:
            self.extensions.append("rocm_docs.doxygen")

        self.doxysphinx_enabled = True

    def setup(self) -> None:
        """Set up default RTD variables."""
        self.extensions.append("rocm_docs")
        full_project_name = self._project_name
        self.html_title = full_project_name
        self.html_theme = "rocm_docs_theme"


__all__ = ["setup", "ROCmDocs"]
