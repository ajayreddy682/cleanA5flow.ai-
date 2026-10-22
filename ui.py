"""Reusable UI pieces for the CleanFlow AI Streamlit application.

Every value interpolated into custom HTML is escaped. Uploaded filenames, cell
values, and category labels are untrusted input and must never be injected raw
into a Markdown/HTML block.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

from utils.validators import format_bytes

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def render_html(content: str) -> None:
    """Render trusted static HTML without Markdown's blank-line parsing issues."""

    st.html(content)


def inject_styles() -> None:
    css_path = ASSETS_DIR / "styles.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        css = ""
    render_html(f"<style>{css}</style>")


def sidebar_brand() -> None:
    render_html(
        """
        <div class="cf-brand">
          <div class="cf-logo">CF</div>
          <div>
            <div class="cf-brand-name">CleanFlow AI</div>
            <div class="cf-brand-sub">Data cleaning &amp; analytics</div>
          </div>
        </div>
        """
    )


def page_header(kicker: str, title: str, subtitle: str = "", badges: Iterable[str] = ()) -> None:
    badge_html = "".join(
        f'<span class="cf-badge blue"><span class="cf-dot"></span>{esc(badge)}</span>'
        for badge in badges
    )
    render_html(
        f"""
        <div class="cf-page-head">
          <div class="cf-page-kicker">{esc(kicker)}</div>
          <h2 class="cf-page-title">{esc(title)}</h2>
          <div class="cf-page-subtitle">{esc(subtitle)}</div>
          {f'<div class="cf-chip-row" style="margin-top:12px">{badge_html}</div>' if badge_html else ''}
        </div>
        """,
    )


def hero(kicker: str, title_html: str, subtitle: str, chips: Iterable[str]) -> None:
    """Render the home hero.

    ``title_html`` is intentionally allowed to contain the ``accent`` span built
    by the caller; it must not contain user data.
    """

    chip_html = "".join(f'<span class="chip">{esc(chip)}</span>' for chip in chips)
    render_html(
        f"""
        <div class="cf-hero">
          <span class="eyebrow"><span class="dot"></span>{esc(kicker)}</span>
          <h1>{title_html}</h1>
          <p class="lede">{esc(subtitle)}</p>
          <div class="chips">{chip_html}</div>
        </div>
        """,
    )


def section_header(title: str, subtitle: str = "") -> None:
    render_html(
        f"""
        <div class="cf-section">
          <div class="cf-section-title">{esc(title)}</div>
          {f'<div class="cf-section-sub">{esc(subtitle)}</div>' if subtitle else ''}
        </div>
        """,
    )


def card_open(extra_class: str = "") -> None:
    render_html(f'<div class="cf-card {esc(extra_class)}">')


def card_close() -> None:
    render_html("</div>")


def kpi_grid(kpis: Iterable[Any], *, empty_label: str = "—") -> None:
    cards = []
    for kpi in kpis:
        if isinstance(kpi, dict):
            label = kpi.get("label", "")
            value = kpi.get("display", kpi.get("value", empty_label))
            hint = kpi.get("hint", "")
        else:
            label = getattr(kpi, "label", "")
            value = getattr(kpi, "display", getattr(kpi, "value", empty_label))
            hint = getattr(kpi, "hint", "")
        value_text = "" if value is None else str(value)
        small = " small" if len(value_text) > 15 else ""
        cards.append(
            f"""
            <div class="cf-kpi">
              <div class="label">{esc(label)}</div>
              <div class="value{small}">{esc(value_text or empty_label)}</div>
              {f'<div class="hint">{esc(hint)}</div>' if hint else ''}
            </div>
            """
        )
    render_html(f'<div class="cf-kpi-grid">{"".join(cards)}</div>')


def metric_strip(items: Iterable[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="cf-stat"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div></div>'
        for label, value in items
    )
    render_html(f'<div class="cf-metric-row">{cells}</div>')


def callout(title: str, body: str, tone: str = "info", icon: str | None = None) -> None:
    icons = {"info": "i", "cyan": "→", "warn": "!", "danger": "×", "success": "✓"}
    symbol = icon if icon is not None else icons.get(tone, "i")
    render_html(
        f"""
        <div class="cf-callout {esc(tone)}">
          <div class="icon">{esc(symbol)}</div>
          <div class="body">
            <strong>{esc(title)}</strong>
            <p>{esc(body)}</p>
          </div>
        </div>
        """,
    )


def empty_state(glyph: str, title: str, body: str) -> None:
    render_html(
        f"""
        <div class="cf-empty">
          <div class="glyph">{esc(glyph)}</div>
          <h4>{esc(title)}</h4>
          <p>{esc(body)}</p>
        </div>
        """,
    )


def feature_grid(items: Iterable[tuple[str, str, str]], *, columns: int = 3) -> None:
    cells = "".join(
        f"""
        <div class="cf-feature">
          <div class="ico">{esc(icon)}</div>
          <h4>{esc(title)}</h4>
          <p>{esc(body)}</p>
        </div>
        """
        for icon, title, body in items
    )
    extra = " cols-2" if columns == 2 else ""
    render_html(f'<div class="cf-feature-grid{extra}">{cells}</div>')


def step_grid(items: Iterable[tuple[str, str, str]]) -> None:
    cells = "".join(
        f"""
        <div class="cf-step">
          <div class="num">{esc(number)}</div>
          <h4>{esc(title)}</h4>
          <p>{esc(body)}</p>
        </div>
        """
        for number, title, body in items
    )
    render_html(f'<div class="cf-step-grid">{cells}</div>')


def badge(text: str, tone: str = "blue") -> str:
    return f'<span class="cf-badge {esc(tone)}">{esc(text)}</span>'


def data_chips(items: Iterable[tuple[str, str]]) -> None:
    chips = "".join(
        f'<span class="cf-data-chip">{esc(label)} <strong>{esc(value)}</strong></span>'
        for label, value in items
    )
    render_html(f'<div class="cf-chip-row">{chips}</div>')


def file_summary_card(
    filename: str,
    file_type: str,
    size_bytes: int,
    tables: int,
    rows: int,
    columns: int,
    encoding: str | None = None,
) -> None:
    render_html(
        f"""
        <div class="cf-card cf-card-tight">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap">
            <div style="display:flex;gap:12px;align-items:center;min-width:0">
              <div class="cf-logo" style="width:38px;height:38px;border-radius:11px;font-size:13px">
                {esc(file_type.upper()[:4])}
              </div>
              <div style="min-width:0">
                <div style="font-weight:750;color:#0b1f3a;font-size:13.5px;overflow-wrap:anywhere">{esc(filename)}</div>
                <div style="font-size:11.5px;color:#64748b;margin-top:2px">
                  {esc(format_bytes(size_bytes))} · {esc(file_type.upper())}
                  {f" · {esc(encoding)}" if encoding else ""}
                </div>
              </div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
              {badge(f"{tables} table" + ("s" if tables != 1 else ""), "blue")}
              {badge(f"{rows:,} rows", "grey")}
              {badge(f"{columns:,} columns", "grey")}
            </div>
          </div>
        </div>
        """,
    )


def sidebar_dataset(
    filename: str | None,
    file_type: str | None,
    rows: int,
    columns: int,
    tables: int,
    cleaned: bool,
) -> None:
    if not filename:
        body = """
        <div class="cf-side-card">
          <div class="label">Active dataset</div>
          <div style="font-size:12px;color:#64748b;margin-top:8px;line-height:1.5">
            No file loaded. Upload a workbook, CSV, JSON, or PDF to begin.
          </div>
        </div>
        """
    else:
        status = badge("Cleaned", "green") if cleaned else badge("Ready to clean", "amber")
        body = f"""
        <div class="cf-side-card">
          <div class="label">Active dataset</div>
          <div class="cf-side-file">
            <div class="icon">{esc((file_type or "?")[:3])}</div>
            <div style="min-width:0">
              <div class="name">{esc(filename)}</div>
              <div class="meta">{rows:,} rows · {columns:,} cols · {tables} table{"s" if tables != 1 else ""}</div>
            </div>
          </div>
          <div style="margin-top:9px">{status}</div>
        </div>
        """
    render_html(body)


def privacy_note() -> None:
    render_html(
        """
        <div class="cf-side-note">
          <strong style="color:#475569">Privacy by default.</strong><br>
          Files are processed in memory for this session. CleanFlow never publishes
          uploads to a web folder. History stores metadata only.
        </div>
        """,
    )


def format_timestamp(value: Any, *, include_time: bool = True) -> str:
    if not value:
        return "—"
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime(
            "%d %b %Y, %H:%M UTC" if include_time else "%d %b %Y"
        )
    except (ValueError, TypeError):
        return str(value)[:40]


def pretty_type(file_type: str) -> str:
    return {
        "xlsx": "Excel workbook",
        "xls": "Legacy Excel",
        "csv": "CSV text",
        "json": "JSON data",
        "pdf": "PDF document",
    }.get(str(file_type).lower(), str(file_type).upper())


def safe_code_language(language: str) -> str:
    allowed = {"python", "dax", "json", "markdown", "text", "sql", "yaml"}
    return language if language in allowed else "text"
