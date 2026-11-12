"""Run reporting: SVG figures and a standalone HTML summary."""

from vcdd.report.figures import build_all as build_figures
from vcdd.report.html import build_report, render_html

__all__ = ["build_report", "render_html", "build_figures"]
