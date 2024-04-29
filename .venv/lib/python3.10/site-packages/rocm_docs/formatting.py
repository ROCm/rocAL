"""Utilities for formatting text."""

from __future__ import annotations

from typing import Any

import re
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path


class Formatter:
    """Formatting class for string substitution and comments parsing."""

    @dataclass
    class _Replacement:
        loc: tuple[int, int]
        text: str

    def __init__(self, context: dict[str, Any]):
        """Initialize Formatter."""
        self.directive_pattern: re.Pattern[str] = re.compile(
            r"(?P<prefix>\$)?\{(?P<directive>[a-zA-z][a-zA-Z0-9_]+)"
            r"(?:\:(?P<parameter>[a-zA-Z0-9_\-\.]+))?\}"
        )
        self.context = context

    def _format_simple(
        self, directive: str, parameter: str | None, loc: tuple[int, int]
    ) -> _Replacement | None:
        # Cannot have a parameter
        if parameter is not None:
            return None
        return self._Replacement(loc, self.context[directive])

    def _format_project(
        self, _: str, parameter: str | None, loc: tuple[int, int]
    ) -> _Replacement | None:
        # Parameter is required
        if parameter is None:
            return None
        if parameter not in self.context["projects"]:
            return None
        return self._Replacement(loc, self.context["projects"][parameter])

    def _format_directive(self, match: re.Match[str]) -> _Replacement | None:
        # As a special case allow `{branch}` and `url` to alias `${branch}`
        # and '${url}' respectively for backwards compatibility.
        # Otherwise the '$' is required
        if match["prefix"] is None and match["directive"] not in [
            "branch",
            "url",
        ]:
            return None

        if match["directive"] in ["branch", "url"]:
            return self._format_simple(
                match["directive"], match["parameter"], match.span()
            )

        if match["directive"] == "project":
            return self._format_project(
                match["directive"], match["parameter"], match.span()
            )

        return None

    def _replacements(self, line: str) -> Generator[_Replacement, None, None]:
        for match in self.directive_pattern.finditer(line):
            replacement = self._format_directive(match)
            if replacement is not None:
                yield replacement

    def format_line(self, line: str) -> str:
        """Substitute variable references into line.

        References of the form ${<variable>} and ${directive:param}
        are substituted
        >>> f = Formatter(
        ...     {
        ...         "branch": "develop",
        ...         "url": "https://example.com",
        ...         "projects": {"project": "https://project.com"},
        ...     }
        ... )
        >>> f.format_line('my branch is ${branch}, {branch} also works')
        'my branch is develop, develop also works'
        >>> f.format_line('Url: ${url} or {url}')
        'Url: https://example.com or https://example.com'
        >>> f.format_line('- url: ${project:project}')
        '- url: https://project.com'

        Unknown references are not replaced.
        >>> f.format_line('{invalid}')
        '{invalid}'
        """
        result: str = ""
        end: int = 0
        for replacement in self._replacements(line):
            assert replacement.loc[0] >= end
            result += line[end : replacement.loc[0]] + replacement.text
            end = replacement.loc[1]

        result += line[end:]
        return result

    def skip_comments(self, lines: Iterable[str]) -> Generator[str, None, None]:
        """Returns a sequence that skips lines as long as they start with '#'.

        Lines after the first "non-comment" line are returned as-is.
        >>> f = Formatter({})
        >>> for l in f.skip_comments(
        ...     ["#comment 0", "# comment 1", "text", "# will not be skipped"]
        ... ):
        ...     print(l)
        text
        # will not be skipped
        """
        iterator = iter(lines)
        for line in iterator:
            if not line.startswith("#"):
                yield line
                break
        yield from iterator


def format_toc(
    input_path: Path, output_path: Path, context: dict[str, Any]
) -> None:
    """Format the input table of contents with additional information."""
    formatter = Formatter(context)
    with (
        open(input_path, encoding="utf-8") as toc_in,
        open(output_path, "w", encoding="utf-8") as toc_out,
    ):
        for line in formatter.skip_comments(toc_in):
            toc_out.write(formatter.format_line(line))
