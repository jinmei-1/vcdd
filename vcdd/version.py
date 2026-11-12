"""Single source of truth for the package version.

``pyproject.toml`` reads this file (``[tool.hatch.version]``), so the version is
declared in exactly one place and the installed package and the distribution
metadata cannot drift apart.
"""

__version__ = "0.3.0"
