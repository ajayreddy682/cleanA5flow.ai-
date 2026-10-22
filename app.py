"""CleanFlow AI — Data Cleaning & Analytics Platform.

Run locally on Windows:
    .venv\\Scripts\\streamlit run app.py

The application is intentionally local-first: uploads are processed in memory,
no paid AI API is required, and cleaning history stores metadata only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from modules import ui
from modules.data_cleaner import (
    CleaningOptions,
    CleaningResult,
    clean_dataframe,
)
from modules.dashboard_generator import (
    AGGREGATIONS,
    CHART_TYPES,
    FilterState,
    apply_filters,
    build_auto_charts,
    build_chart,
    empty_figure,
    filterable_columns,
    filtered_summary,
)
from modules.data_profiler import (
    ColumnKind,
    DataProfile,
    format_metric,
    format_number,
    profile_dataframe,
    recommend_charts,
    recommend_kpis,
)
from modules.file_loader import FileLoadError, LoadedFile, TableData, load_uploaded_file
from modules.history_store import (
    clear_jobs,
    create_job,
    export_history_csv,
    history_frame,
    history_stats,
    load_jobs,
)
from modules.powerbi_exporter import build_powerbi_package, prepare_model
from modules.report_generator import (
    build_activity_log_csv,
    build_cleaned_csv,
    build_cleaned_workbook,
    build_quality_report,
    dataframe_to_csv,
)
from utils.security import safe_download_filename, safe_error_message
from utils.validators import (
    MAX_FILE_SIZE_BYTES,
    UPLOADER_TYPES,
    ValidationError,
    format_bytes,
    sanitize_filename,
)

LOGGER = logging.getLogger("cleanflow.app")

APP_TITLE = "CleanFlow AI"
APP_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
NAV_ITEMS = ["Home", "Upload & Clean", "Analytics", "Power BI", "History"]

SAMPLE_FILES = [
    ("Sample sales", "sample_sales.csv", "Messy orders with duplicates, currency, dates, and categories."),
    ("Messy customers", "sample_messy_customers.csv", "IDs with leading zeros, inconsistent casing, and gaps."),
    ("Nested orders", "sample_orders.json", "A JSON API-style export with two record arrays."),
]

st.set_page_config(
    page_title="CleanFlow AI — Data Cleaning & Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults: dict[str, Any] = {
        "page": "Home",
        "nav_widget": "Home",
        "nav_pending": None,
        "loaded": None,
        "upload_signature": None,
        "attempted_signature": None,
        "failed_attempt": None,
        "results": {},
        "clean_signature": None,
        "clean_options": None,
        "clean_timestamp": None,
        "active_table": None,
        "raw_profiles": {},
        "filter_profiles": {},
        "download_bundle_key": None,
        "download_bundle": None,
        "powerbi_key": None,
        "powerbi_model": None,
        "powerbi_package": None,
        "powerbi_package_key": None,
        "last_job_id": None,
        "flash": [],
        "filter_token": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def queue_flash(tone: str, title: str, body: str = "") -> None:
    st.session_state.flash.append((tone, title, body))


def render_flashes() -> None:
    items = st.session_state.get("flash", [])
    if not items:
        return
    st.session_state.flash = []
    for tone, title, body in items[:3]:
        ui.callout(title, body or " ", tone=tone)


def reset_dataset(*, navigate_home: bool = False) -> None:
    st.session_state.loaded = None
    st.session_state.upload_signature = None
    st.session_state.attempted_signature = None
    st.session_state.failed_attempt = None
    st.session_state.results = {}
    st.session_state.clean_signature = None
    st.session_state.clean_options = None
    st.session_state.clean_timestamp = None
    st.session_state.active_table = None
    st.session_state.raw_profiles = {}
    st.session_state.filter_profiles = {}
    st.session_state.download_bundle = None
    st.session_state.download_bundle_key = None
    st.session_state.powerbi_key = None
    st.session_state.powerbi_model = None
    st.session_state.powerbi_package = None
    st.session_state.powerbi_package_key = None
    if navigate_home:
        st.session_state.page = "Home"
        st.session_state.nav_pending = "Home"


def _loaded() -> LoadedFile | None:
    return st.session_state.get("loaded")


def _active_result() -> CleaningResult | None:
    results: dict[str, CleaningResult] = st.session_state.get("results") or {}
    if not results:
        return None
    name = st.session_state.get("active_table")
    if name in results:
        return results[name]
    first = next(iter(results))
    st.session_state.active_table = first
    return results[first]


def _nav_changed() -> None:
    st.session_state.page = st.session_state.nav_widget


def render_sidebar() -> None:
    with st.sidebar:
        ui.sidebar_brand()
        ui.render_html('<div class="cf-side-label">Workspace</div>')
        pending = st.session_state.get("nav_pending")
        if pending in NAV_ITEMS:
            # Apply programmatic navigation before the radio widget exists in
            # this run. Streamlit forbids changing a widget key afterwards.
            st.session_state.nav_widget = pending
            st.session_state.nav_pending = None
        elif st.session_state.get("nav_widget") not in NAV_ITEMS:
            st.session_state.nav_widget = st.session_state.get("page", "Home")
        st.radio(
            "Navigation",
            NAV_ITEMS,
            key="nav_widget",
            label_visibility="collapsed",
            on_change=_nav_changed,
        )

        loaded: LoadedFile | None = _loaded()
        results: dict[str, CleaningResult] = st.session_state.get("results") or {}
        active = _active_result() if results else None
        if loaded:
            rows = len(active.cleaned) if active else loaded.total_rows
            columns = active.cleaned.shape[1] if active else loaded.primary_table.columns
            ui.sidebar_dataset(
                loaded.filename,
                loaded.file_type,
                rows,
                columns,
                len(loaded.tables),
                bool(results),
            )
            if results:
                ui.render_html("<div style='height:6px'></div>")
                if st.button("Open analytics", key="side_analytics", use_container_width=True):
                    go_to("Analytics")
            if st.button("Clear current dataset", key="side_clear", use_container_width=True):
                reset_dataset(navigate_home=True)
                queue_flash("info", "Dataset cleared", "Uploaded data and generated results were removed from this session.")
                st.rerun()
        else:
            ui.sidebar_dataset(None, None, 0, 0, 0, False)
        ui.privacy_note()
        ui.render_html(
            f'<div class="cf-footnote">CleanFlow AI v{APP_VERSION} · Local-first · No paid AI API</div>'
        )


def go_to(page: str) -> None:
    st.session_state.page = page
    st.session_state.nav_pending = page
    st.rerun()


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
def _signature(name: str, data: bytes) -> str:
    return hashlib.sha256(data).hexdigest() + ":" + name


def _file_type_from_name(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".") or "unknown"


def _record_failed_job(name: str, data: bytes, message: str, started: float) -> None:
    try:
        create_job(
            filename=sanitize_filename(name, "upload"),
            file_type=_file_type_from_name(name),
            size_bytes=len(data),
            status="failed",
            error=message,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception:
        LOGGER.exception("Could not record failed job")


def _apply_loaded(loaded: LoadedFile, signature: str, *, navigate: bool) -> None:
    st.session_state.loaded = loaded
    st.session_state.upload_signature = signature
    st.session_state.attempted_signature = signature
    st.session_state.failed_attempt = None
    st.session_state.results = {}
    st.session_state.clean_signature = None
    st.session_state.clean_options = None
    st.session_state.clean_timestamp = None
    st.session_state.active_table = loaded.primary_table.name
    st.session_state.raw_profiles = {}
    st.session_state.filter_profiles = {}
    st.session_state.download_bundle = None
    st.session_state.download_bundle_key = None
    st.session_state.powerbi_key = None
    st.session_state.powerbi_model = None
    st.session_state.powerbi_package = None
    st.session_state.powerbi_package_key = None
    st.session_state.last_job_id = None
    if navigate:
        st.session_state.page = "Upload & Clean"
        st.session_state.nav_pending = "Upload & Clean"
    queue_flash(
        "success",
        f"Loaded {loaded.filename}",
        f"{len(loaded.tables)} table{'s' if len(loaded.tables) != 1 else ''}, "
        f"{loaded.total_rows:,} total rows. Review the quality snapshot, then clean.",
    )


def process_file_bytes(
    name: str,
    data: bytes,
    *,
    attempt_ocr: bool = False,
    navigate: bool = False,
    show_progress: bool = True,
) -> bool:
    """Validate, parse, and make an upload active.

    Returns True on success. All failures are shown as safe messages and
    recorded as metadata-only history entries.
    """

    signature = _signature(name, data)
    st.session_state.attempted_signature = signature
    progress = st.progress(8, text="Validating file type, signature, and size…") if show_progress else None
    started = time.perf_counter()
    try:
        if progress:
            progress.progress(28, text="Reading tables and detecting structure…")
        loaded = load_uploaded_file(name, data, attempt_ocr=attempt_ocr)
        if progress:
            progress.progress(92, text="Checking table limits…")
        _apply_loaded(loaded, signature, navigate=navigate)
        if progress:
            progress.progress(100, text="Ready")
            progress.empty()
        return True
    except (ValidationError, FileLoadError) as exc:
        message = str(exc)
        if progress:
            progress.empty()
        st.session_state.failed_attempt = {
            "name": sanitize_filename(name, "upload"),
            "file_type": _file_type_from_name(name),
            "signature": signature,
            "message": message,
            "data": data,
        }
        _record_failed_job(name, data, message, started)
        queue_flash("danger", "This file could not be loaded", message)
        return False
    except Exception as exc:
        message = safe_error_message(exc)
        if progress:
            progress.empty()
        st.session_state.failed_attempt = {
            "name": sanitize_filename(name, "upload"),
            "file_type": _file_type_from_name(name),
            "signature": signature,
            "message": message,
            "data": data,
        }
        _record_failed_job(name, data, message, started)
        queue_flash("danger", "This file could not be loaded", message)
        return False


def handle_uploader(uploaded: Any, *, navigate: bool) -> None:
    if uploaded is None:
        return
    try:
        data = uploaded.getvalue()
    except Exception:
        queue_flash("danger", "The upload could not be read", "Please try selecting the file again.")
        return
    signature = _signature(uploaded.name, data)
    if st.session_state.get("upload_signature") == signature and _loaded() is not None:
        return
    if st.session_state.get("attempted_signature") == signature:
        # A previous attempt with these exact bytes was already made. Use the
        # explicit retry control (or select a different file) rather than
        # retrying on every Streamlit rerun.
        return
    process_file_bytes(
        uploaded.name,
        data,
        attempt_ocr=bool(st.session_state.get("ocr_enabled", False)),
        navigate=navigate,
    )
    st.rerun()


def load_sample(filename: str) -> None:
    path = SAMPLE_DIR / filename
    try:
        data = path.read_bytes()
    except OSError:
        queue_flash("danger", "Sample not found", "The sample data file is missing from this installation.")
        st.rerun()
        return
    process_file_bytes(path.name, data, navigate=True)
    st.rerun()


def render_retry_upload() -> None:
    failed = st.session_state.get("failed_attempt")
    if not failed:
        return
    ui.render_html("<div style='height:8px'></div>")
    cols = st.columns([1.2, 1.2, 3])
    with cols[0]:
        if st.button("Retry with OCR" if failed.get("file_type") == "pdf" else "Retry", key="retry_upload", use_container_width=True):
            st.session_state.attempted_signature = None
            succeeded = process_file_bytes(
                failed["name"],
                failed["data"],
                attempt_ocr=True,
                navigate=False,
            )
            if succeeded:
                st.session_state.page = "Upload & Clean"
                st.session_state.nav_pending = "Upload & Clean"
            st.rerun()
    with cols[1]:
        if st.button("Dismiss error", key="dismiss_failed", use_container_width=True):
            st.session_state.attempted_signature = failed.get("signature")
            st.session_state.failed_attempt = None
            st.rerun()
    with cols[2]:
        st.caption(
            "Retry uses the same bytes from this session. OCR requires the optional local Tesseract/Poppler setup."
        )


def render_uploader_block(
    *,
    key: str,
    navigate: bool,
    heading: str = "Drop your file here",
    show_ocr_toggle: bool = True,
) -> None:
    if show_ocr_toggle:
        st.checkbox(
            "Try OCR if this PDF is scanned",
            key="ocr_enabled",
            help=(
                "Optional. Requires Tesseract OCR, Poppler, pytesseract, and pdf2image. "
                "See requirements-ocr.txt and the README. CleanFlow does not run OCR by default."
            ),
        )
    uploaded = st.file_uploader(
        heading,
        type=UPLOADER_TYPES,
        accept_multiple_files=False,
        key=key,
        help=f"Excel (.xlsx/.xls), CSV, JSON, or PDF. Maximum {format_bytes(MAX_FILE_SIZE_BYTES)} per file.",
    )
    st.caption(
        f"Drag and drop is supported · Max {format_bytes(MAX_FILE_SIZE_BYTES)} · "
        "Uploads stay in this Streamlit session."
    )
    handle_uploader(uploaded, navigate=navigate)
    render_retry_upload()


def render_sample_buttons(prefix: str) -> None:
    st.caption("No file handy? Explore a small, intentionally messy sample.")
    columns = st.columns(3)
    for column, (label, filename, description) in zip(columns, SAMPLE_FILES):
        with column:
            if st.button(label, key=f"{prefix}_{filename}", use_container_width=True):
                load_sample(filename)
            st.caption(description)


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
def render_home() -> None:
    ui.hero(
        "Local-first data operations",
        'Messy data in.<br><span class="accent">Decision-ready data out.</span>',
        "Clean messy business files, review every transformation, and export analysis-ready "
        "Excel, CSV, and Power BI datasets — without a paid AI API.",
        ["Excel", "CSV", "JSON", "PDF", "Power BI-ready", "No API keys"],
    )

    actions = st.columns([1.1, 1.1, 2.5])
    with actions[0]:
        if st.button("Upload data", type="primary", use_container_width=True, key="home_cta_upload"):
            go_to("Upload & Clean")
    with actions[1]:
        if st.button("Try sample data", use_container_width=True, key="home_cta_sample"):
            load_sample("sample_sales.csv")
    with actions[2]:
        st.caption("The sample loads real messy data so you can inspect the full workflow before using your own file.")

    jobs = load_jobs()
    stats = history_stats(jobs)
    ui.section_header(
        "Your workspace at a glance",
        "Live counts from this installation. No invented customer numbers, reviews, or usage claims.",
    )
    ui.kpi_grid(
        [
            {"label": "Cleaning jobs", "display": format_number(stats["total_jobs"]), "hint": "Completed and failed attempts"},
            {"label": "Rows cleaned", "display": format_number(stats["total_rows_cleaned"]), "hint": "Rows in completed cleaned outputs"},
            {"label": "Duplicates removed", "display": format_number(stats["duplicates_removed"]), "hint": "Exact duplicates across all columns"},
            {
                "label": "Last activity",
                "display": ui.format_timestamp(stats["last_run"]) if stats["last_run"] else "No jobs yet",
                "hint": "Times shown in UTC",
            },
        ]
    )

    ui.render_html("<div style='height:18px'></div>")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        ui.section_header("Start a cleaning job", "Upload a workbook, CSV, JSON file, or a digital PDF.")
        render_uploader_block(key="home_uploader", navigate=True)
        ui.render_html("<div style='height:8px'></div>")
        render_sample_buttons("home_sample")
    with right:
        ui.section_header("What you get")
        ui.feature_grid(
            [
                ("01", "Auditable cleaning", "Every trim, type conversion, and removal is logged with counts."),
                ("02", "Excel + CSV exports", "Professional workbooks with filters, frozen headers, and tables."),
                ("03", "Interactive charts", "KPIs and Plotly visuals generated only when the data supports them."),
                ("04", "Power BI package", "Star-schema suggestions, DAX, data dictionary, and import guide."),
            ],
            columns=2,
        )
        ui.callout(
            "Preview before you download",
            "Cleaning runs on a copy. You review the proposed changes, quality report, and audit log before exporting.",
            tone="cyan",
        )

    ui.section_header("How it works", "Three deliberate steps — no black-box transformations.")
    ui.step_grid(
        [
            ("STEP 01", "Load and inspect", "Validate the file, choose a worksheet or extracted PDF table, and review the raw quality snapshot."),
            ("STEP 02", "Clean with controls", "Pick text, type, duplicate, and empty-row options. Problem values are flagged, never invented."),
            ("STEP 03", "Analyse and export", "Explore interactive charts, then download Excel, CSV, a quality report, and a Power BI package."),
        ]
    )

    ui.section_header("Recent processing jobs", "Stored as metadata only. File contents and cell values are never saved to history.")
    if jobs:
        recent = jobs[:5]
        st.dataframe(
            history_frame(recent),
            width="stretch",
            hide_index=True,
            height=min(220, 44 + len(recent) * 32),
        )
        if st.button("View full history", key="home_view_history"):
            go_to("History")
    else:
        ui.empty_state(
            "—",
            "No processing jobs yet",
            "Upload a file above or load a sample. Your first completed cleaning run will appear here.",
        )

    ui.section_header("Built for everyday business data", "Small businesses, schools, clinics, shops, accountants, students, and analysts.")
    ui.feature_grid(
        [
            ("XL", "Excel workbooks", "Multiple worksheets, filters, frozen headers, and formats that respect dates and numbers."),
            ("CSV", "CSV imports", "Delimiter detection, encoding fallback, header detection, and leading-zero preservation."),
            ("JSON", "JSON and PDF", "Flatten record arrays into tables and extract PDF tables or line-based text when the structure allows."),
            ("BI", "Power BI prep", "Fact/dimension suggestions, relationship guidance, DAX measures, and a data dictionary."),
            ("PR", "Privacy-aware", "In-memory processing, size limits, formula protection, and metadata-only history."),
            ("AU", "Audit trail", "Every operation is written to a report with affected rows, columns, and cells."),
        ]
    )

    ui.callout(
        "Honest scope",
        "CleanFlow does not generate arbitrary .pbix report files and does not run OCR unless you install the optional tools. "
        "The in-app dashboard is a Plotly dashboard, not a Power BI report. No paid AI service is required for the core workflow.",
        tone="warn",
        icon="!",
    )


# ---------------------------------------------------------------------------
# Upload & Clean
# ---------------------------------------------------------------------------
def render_upload_clean() -> None:
    ui.page_header(
        "Upload & clean",
        "Inspect, configure, and clean",
        "Review the raw table and quality snapshot first. Cleaning options are explicit, and nothing is overwritten.",
        badges=["Preview first", "Auditable", "Original preserved"],
    )
    loaded = _loaded()
    if loaded is None:
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            ui.section_header("Choose a file", "Drag and drop, or browse. File type and content are both validated.")
            render_uploader_block(key="upload_uploader", navigate=False)
            render_sample_buttons("upload_sample")
        with right:
            ui.section_header("Supported input")
            ui.feature_grid(
                [
                    ("XLSX", "Excel", "Multiple worksheets; select the table to inspect."),
                    ("CSV", "CSV", "Encoding and delimiter detection with clear errors."),
                    ("JSON", "JSON", "Record arrays, nested objects, JSON Lines."),
                    ("PDF", "PDF", "Digital tables and text; scanned pages need optional OCR."),
                ]
            )
            ui.callout(
                "Scanned PDFs need OCR",
                "If a PDF has no text layer, CleanFlow explains that OCR is required. An optional free pathway is "
                "available through Tesseract and Poppler; install requirements-ocr.txt and follow the README.",
                tone="warn",
            )
        return

    active_table = loaded.table_by_name(st.session_state.get("active_table") or "") or loaded.primary_table
    st.session_state.active_table = active_table.name

    ui.file_summary_card(
        loaded.filename,
        loaded.file_type,
        loaded.size_bytes,
        len(loaded.tables),
        loaded.total_rows,
        loaded.primary_table.columns,
        loaded.encoding,
    )

    displayed_warnings = list(dict.fromkeys(loaded.warnings + active_table.warnings))
    if displayed_warnings:
        with st.expander(f"File notes ({len(displayed_warnings)})", expanded=False):
            for warning in displayed_warnings[:12]:
                st.markdown(f"- {warning}")

    names = [table.name for table in loaded.tables]
    if st.session_state.get("active_table") not in names:
        st.session_state.active_table = names[0]
    if len(names) > 1:
        st.selectbox(
            "Worksheet / extracted table",
            names,
            key="active_table",
            format_func=lambda name: next(
                (table.label for table in loaded.tables if table.name == name), name
            ),
        )
        active_table = loaded.table_by_name(st.session_state.active_table) or loaded.primary_table
    else:
        st.caption(f"Table: **{active_table.name}** · {active_table.rows:,} rows × {active_table.columns:,} columns")

    raw_profile = get_raw_profile(active_table)
    raw = active_table.dataframe
    empty_rows = int(raw.isna().all(axis=1).sum()) if raw.shape[1] else 0
    empty_cols = int(raw.isna().all(axis=0).sum()) if raw.shape[1] else 0

    ui.section_header("Raw quality snapshot", "Computed on the selected table before any cleaning.")
    ui.kpi_grid(
        [
            {"label": "Rows", "display": format_number(raw_profile.rows), "hint": "Before cleaning"},
            {"label": "Columns", "display": format_number(raw_profile.columns), "hint": "Before cleaning"},
            {"label": "Missing cells", "display": format_number(raw_profile.missing_cells), "hint": f"{raw_profile.missing_pct:.1f}% of cells"},
            {"label": "Exact duplicate rows", "display": format_number(raw_profile.duplicate_rows), "hint": "Across all columns"},
            {"label": "Empty rows", "display": format_number(empty_rows), "hint": "All cells empty"},
            {"label": "Empty columns", "display": format_number(empty_cols), "hint": "All cells empty"},
        ]
    )

    preview_tabs = st.tabs(["Preview", "Column types", "File notes"])
    with preview_tabs[0]:
        st.caption(f"Showing {min(150, len(raw)):,} of {len(raw):,} raw rows. No values have been changed.")
        st.dataframe(raw.head(150), width="stretch", hide_index=True, height=360)
    with preview_tabs[1]:
        raw_frame = raw_profile.to_frame()
        display = raw_frame[
            ["column", "type", "role", "missing", "missing_pct", "unique", "sample", "issues"]
        ].rename(
            columns={
                "column": "Column",
                "type": "Detected type",
                "role": "Likely role",
                "missing": "Missing",
                "missing_pct": "Missing %",
                "unique": "Unique",
                "sample": "Examples",
                "issues": "Notes",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True, height=340)
        st.caption("Type labels in this snapshot are preview-only. The cleaning report shows the final converted types.")
    with preview_tabs[2]:
        if displayed_warnings:
            for warning in displayed_warnings:
                st.markdown(f"- {warning}")
        else:
            st.write("No file-specific warnings were detected.")
        st.caption("Formulas and macros are never executed. Excel reads cached values only.")

    ui.render_html('<div class="cf-divider"></div>')
    ui.section_header("Cleaning options", "Choose what should change. The original file stays untouched.")
    render_cleaning_form(loaded, active_table)


def get_raw_profile(table: TableData) -> DataProfile:
    cache: dict[str, DataProfile] = st.session_state.setdefault("raw_profiles", {})
    key = f"{st.session_state.get('upload_signature', '')}::{table.name}"
    if key not in cache:
        with st.spinner("Profiling the raw table…"):
            cache[key] = profile_dataframe(table.dataframe)
    return cache[key]


def parse_missing_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for line in value.splitlines():
        token = line.strip()
        if not token:
            continue
        if token.lower() in {"<blank>", "(blank)", "blank"}:
            token = ""
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def render_cleaning_form(loaded: LoadedFile, active_table: TableData) -> None:
    with st.form("cleaning_form", clear_on_submit=False):
        with st.expander("Text & missing values", expanded=True):
            row = st.columns(2)
            with row[0]:
                standardize = st.checkbox("Standardize column names", value=True, key="opt_standardize", help="Lower-case snake_case, unique names.")
                trim = st.checkbox("Trim surrounding whitespace", value=True, key="opt_trim")
                normalize_text = st.checkbox("Normalize Unicode, quotes, and repeated spaces", value=True, key="opt_normalize_text")
            with row[1]:
                normalize_missing = st.checkbox("Treat common placeholders as missing", value=True, key="opt_missing")
                normalize_categories = st.checkbox(
                    "Merge category labels that differ only by case",
                    value=False,
                    key="opt_categories",
                    help="Uses the most frequent spelling as the canonical label. Off by default so labels are not changed silently.",
                )
                st.caption("Missing values are never filled or invented. They remain blank in the output.")
            token_text = st.text_area(
                "Placeholder values to treat as missing (one per line; use <blank> for empty cells)",
                value="n/a\nn.a.\nnull\nnone\nnan\nnil\nnot available\n<blank>",
                height=150,
                key="opt_tokens",
                help="'NA' is deliberately not included by default because it can be a real category or state code.",
            )

        with st.expander("Types, numbers & dates", expanded=True):
            row = st.columns(2)
            with row[0]:
                parse_types = st.checkbox("Detect and convert types", value=True, key="opt_parse_types")
                convert_currency = st.checkbox("Convert currency text to numbers", value=True, key="opt_currency")
                convert_percentage = st.checkbox("Convert percentages to percentage points", value=True, key="opt_percentage")
                parse_dates = st.checkbox("Parse valid dates", value=True, key="opt_dates")
            with row[1]:
                date_order = st.selectbox(
                    "Ambiguous date order",
                    ["Auto (day-first when ambiguous)", "Day first (DD/MM/YYYY)", "Month first (MM/DD/YYYY)"],
                    index=0,
                    key="opt_date_order",
                    help="ISO dates and unambiguous dates are detected directly. This setting controls dates where both readings are possible.",
                )
                decimal_style = st.selectbox(
                    "Decimal separator",
                    ["Auto-detect", "Dot decimal (1,234.56)", "Comma decimal (1.234,56)"],
                    index=0,
                    key="opt_decimal",
                )
                numeric_threshold = st.slider(
                    "Numeric conversion threshold",
                    min_value=50,
                    max_value=100,
                    value=95,
                    step=5,
                    format="%d%%",
                    key="opt_numeric_threshold",
                    help="A column is converted only if at least this share of non-empty values parse. Otherwise it stays text.",
                )
                date_threshold = st.slider(
                    "Date conversion threshold",
                    min_value=50,
                    max_value=100,
                    value=95,
                    step=5,
                    format="%d%%",
                    key="opt_date_threshold",
                    help="A column is converted only if at least this share of non-empty values parse as dates.",
                )

        with st.expander("Duplicates, empty rows & outliers", expanded=True):
            row = st.columns(2)
            with row[0]:
                remove_duplicates = st.checkbox(
                    "Remove exact duplicate rows",
                    value=True,
                    key="opt_duplicates",
                    help="Only rows that match on ALL cleaned columns are duplicates. Matching one ID is never enough.",
                )
                drop_empty_rows = st.checkbox("Remove fully empty rows", value=True, key="opt_empty_rows")
                drop_empty_columns = st.checkbox("Remove fully empty columns", value=True, key="opt_empty_cols")
            with row[1]:
                detect_outliers = st.checkbox(
                    "Flag possible outliers (never remove)",
                    value=True,
                    key="opt_outliers",
                    help="Uses the IQR rule. Outliers are listed in the report and kept in the cleaned data.",
                )
                st.caption(
                    "Destructive operations are recorded with counts in the cleaning report and audit log."
                )

        submitted = st.form_submit_button("Clean data", type="primary", use_container_width=True)

    if submitted:
        options = CleaningOptions(
            standardize_columns=standardize,
            trim_whitespace=trim,
            normalize_unicode=normalize_text,
            collapse_whitespace=normalize_text,
            normalize_missing=normalize_missing,
            missing_tokens=parse_missing_tokens(token_text),
            normalize_category_case=normalize_categories,
            parse_types=parse_types,
            convert_currency=convert_currency,
            convert_percentage=convert_percentage,
            parse_dates=parse_dates,
            day_first={
                "Auto (day-first when ambiguous)": "auto",
                "Day first (DD/MM/YYYY)": "day_first",
                "Month first (MM/DD/YYYY)": "month_first",
            }[date_order],
            decimal_style={
                "Auto-detect": "auto",
                "Dot decimal (1,234.56)": "dot",
                "Comma decimal (1.234,56)": "comma",
            }[decimal_style],
            numeric_threshold=numeric_threshold / 100,
            date_threshold=date_threshold / 100,
            remove_duplicates=remove_duplicates,
            drop_empty_rows=drop_empty_rows,
            drop_empty_columns=drop_empty_columns,
            detect_outliers=detect_outliers,
        )
        run_cleaning(loaded, options)
    else:
        results = st.session_state.get("results") or {}
        if results:
            ui.render_html("<div style='height:14px'></div>")
            result = _active_result()
            if result is not None:
                render_cleaning_result(result)


def run_cleaning(loaded: LoadedFile, options: CleaningOptions) -> None:
    progress = st.progress(0, text="Starting cleaning pipeline…")
    started = time.perf_counter()
    results: dict[str, CleaningResult] = {}
    try:
        total = len(loaded.tables)
        for index, table in enumerate(loaded.tables):
            progress.progress(
                int(index / max(total, 1) * 85),
                text=f"Cleaning {table.name} ({index + 1} of {total})…",
            )
            results[table.name] = clean_dataframe(table.dataframe, options, table_name=table.name)
            progress.progress(
                int((index + 1) / max(total, 1) * 85),
                text=f"Cleaned {table.name}",
            )
        progress.progress(90, text="Finalizing cleaned tables…")
        duration_ms = int((time.perf_counter() - started) * 1000)
        progress.progress(92, text="Storing cleaning results…")
        st.session_state.results = results
        st.session_state.clean_signature = options.fingerprint()
        st.session_state.clean_options = options
        st.session_state.clean_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        st.session_state.active_table = (
            st.session_state.get("active_table")
            if st.session_state.get("active_table") in results
            else next(iter(results))
        )
        st.session_state.download_bundle = None
        st.session_state.download_bundle_key = None
        st.session_state.powerbi_key = None
        st.session_state.powerbi_model = None
        st.session_state.powerbi_package = None
        st.session_state.powerbi_package_key = None
        st.session_state.filter_profiles = {}
        progress.progress(95, text="Recording audit summary…")
        job = create_job(
            filename=loaded.filename,
            file_type=loaded.file_type,
            size_bytes=loaded.size_bytes,
            results=results,
            status="completed",
            duration_ms=duration_ms,
        )
        st.session_state.last_job_id = job.get("id")
        progress.progress(98, text="Finishing cleaning run…")
        progress.progress(100, text="Cleaning complete")
        rows_out = sum(result.summary.rows_after for result in results.values())
        queue_flash(
            "success",
            "Cleaning complete",
            f"{len(results)} table{'s' if len(results) != 1 else ''} cleaned · "
            f"{rows_out:,} output rows · {duration_ms / 1000:.1f}s. Review the report before downloading.",
        )
        st.rerun()
    except Exception as exc:
        progress.empty()
        message = safe_error_message(exc)
        duration_ms = int((time.perf_counter() - started) * 1000)
        create_job(
            filename=loaded.filename,
            file_type=loaded.file_type,
            size_bytes=loaded.size_bytes,
            status="failed",
            error=message,
            duration_ms=duration_ms,
        )
        st.error(message)
    finally:
        # Always remove the progress widget, including when Streamlit reruns
        # or an error occurs while recording the job summary.
        progress.empty()


def _arrow_safe_preview(frame: pd.DataFrame, rows: int = 8) -> pd.DataFrame:
    """Return a display-only preview with mixed Python types normalised.

    Streamlit's Arrow conversion can repeatedly retry on object columns that
    mix values such as integers and strings. Keep the cleaning result intact;
    normalise only the small copy sent to the preview widget.
    """

    preview = frame.head(rows).copy()
    for column in preview.columns:
        series = preview[column]
        if series.dtype != object or series.dropna().map(type).nunique() <= 1:
            continue

        def display_scalar(value: Any) -> Any:
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            return str(value)

        preview[column] = series.map(display_scalar)
    return preview


def render_cleaning_result(result: CleaningResult) -> None:
    summary = result.summary
    options = st.session_state.get("clean_options") or result.options
    cleaned_at = st.session_state.get("clean_timestamp")

    ui.section_header(
        "Cleaning result",
        f"Table: {result.table_name}"
        + (f" · cleaned {ui.format_timestamp(cleaned_at)}" if cleaned_at else ""),
    )
    ui.kpi_grid(
        [
            {"label": "Rows after", "display": format_number(summary.rows_after), "hint": f"From {summary.rows_before:,}"},
            {"label": "Columns after", "display": format_number(summary.columns_after), "hint": f"From {summary.columns_before:,}"},
            {"label": "Duplicates removed", "display": format_number(summary.duplicates_removed), "hint": "Exact, all-column matches"},
            {"label": "Missing cells", "display": format_number(summary.missing_cells_after), "hint": f"Before: {summary.missing_cells_before:,}"},
            {"label": "Values converted", "display": format_number(summary.values_converted), "hint": "Types parsed"},
            {"label": "Values flagged", "display": format_number(summary.invalid_values), "hint": "Invalid or unconvertible"},
            {"label": "Outliers flagged", "display": format_number(summary.outliers_flagged), "hint": "Kept, not removed"},
        ]
    )

    if summary.invalid_values:
        ui.callout(
            "Some values need review",
            f"{summary.invalid_values} value(s) could not be converted within the selected threshold. "
            "They are blank in the cleaned column and their original values are listed in the report and audit trail.",
            tone="warn",
        )
    if summary.category_groups_detected and not options.normalize_category_case:
        ui.callout(
            "Category spelling differences detected",
            f"{summary.category_groups_detected} group(s) of labels differ only by letter case. Nothing was merged. "
            "Review them under Cleaning report, or enable category case merging and re-run.",
            tone="info",
        )

    ui.render_html("<div style='height:6px'></div>")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Before cleaning** (first 8 rows)")
        st.dataframe(_arrow_safe_preview(result.raw), width="stretch", hide_index=True, height=280)
    with right:
        st.markdown("**After cleaning** (first 8 rows)")
        st.dataframe(_arrow_safe_preview(result.cleaned), width="stretch", hide_index=True, height=280)
    st.caption(
        "Before/after samples may not align row-for-row after empty-row and duplicate removal. "
        "The full audit trail records every operation."
    )

    tabs = st.tabs(["Quality report", "Conversions & flags", "Audit log", "Options", "Downloads"])
    with tabs[0]:
        profile_frame = result.profile.to_frame()
        if not profile_frame.empty:
            display = profile_frame[
                ["column", "type", "role", "missing", "missing_pct", "unique", "minimum", "maximum", "mean", "median", "total", "outliers", "issues", "sample"]
            ].rename(
                columns={
                    "column": "Column",
                    "type": "Type",
                    "role": "Role",
                    "missing": "Missing",
                    "missing_pct": "Missing %",
                    "unique": "Unique",
                    "minimum": "Min",
                    "maximum": "Max",
                    "mean": "Mean",
                    "median": "Median",
                    "total": "Total",
                    "outliers": "Outliers",
                    "issues": "Issues",
                    "sample": "Examples",
                }
            )
            st.dataframe(display, width="stretch", hide_index=True, height=380)
        else:
            st.info("No columns remain after cleaning.")
        if result.warnings:
            with st.expander("Notes and limitations", expanded=False):
                for warning in result.warnings:
                    st.markdown(f"- {warning}")
        if result.outliers_frame().shape[0]:
            with st.expander("Outlier examples (not removed)", expanded=False):
                st.dataframe(result.outliers_frame(), width="stretch", hide_index=True)

    with tabs[1]:
        if result.conversions:
            conversions = pd.DataFrame(result.conversions).rename(
                columns={
                    "column": "Column",
                    "from": "From",
                    "to": "To",
                    "values_converted": "Converted",
                    "values_flagged": "Flagged",
                    "notes": "Notes",
                    "examples": "Examples",
                }
            )
            st.dataframe(conversions, width="stretch", hide_index=True, height=280)
        else:
            st.info("No type conversions were applied with these options.")
        if result.invalid_values:
            with st.expander(f"Invalid value details ({len(result.invalid_values)} shown)", expanded=False):
                st.dataframe(pd.DataFrame(result.invalid_values), width="stretch", hide_index=True)
        category_rows = []
        for item in result.category_groups:
            for group in item["groups"]:
                category_rows.append(
                    {
                        "Column": item["column"],
                        "Variants": ", ".join(variant["value"] for variant in group["variants"]),
                        "Counts": ", ".join(str(variant["count"]) for variant in group["variants"]),
                        "Suggested canonical": group["canonical"],
                    }
                )
        if category_rows:
            with st.expander("Inconsistent category labels", expanded=False):
                st.dataframe(pd.DataFrame(category_rows), width="stretch", hide_index=True)
        if not result.duplicates_removed_frame.empty:
            with st.expander("Duplicate rows removed (sample)", expanded=False):
                st.dataframe(result.duplicates_removed_frame.head(100), width="stretch", hide_index=True)
                st.caption("Showing up to 100 of the removed rows. Full count is in the summary.")
        if not result.column_renames:
            st.caption("Column names were already standard or standardization was disabled.")

    with tabs[2]:
        audit = result.audit_frame()
        if audit.empty:
            st.info("No transformations were required with the selected options.")
        else:
            st.dataframe(audit, width="stretch", hide_index=True, height=360)
            st.caption("Timestamps are UTC. Every destructive operation is counted and described.")

    with tabs[3]:
        option_rows = [{"Option": key, "Value": str(value)} for key, value in options.to_dict().items()]
        st.dataframe(pd.DataFrame(option_rows), width="stretch", hide_index=True)
        st.caption("These are the exact options used for this result. Change options above and clean again to apply new settings.")

    with tabs[4]:
        render_downloads(result, key_prefix="clean")

    ui.render_html("<div style='height:10px'></div>")
    actions = st.columns([1.2, 1.2, 2])
    with actions[0]:
        if st.button("Continue to analytics", type="primary", key="to_analytics", use_container_width=True):
            go_to("Analytics")
    with actions[1]:
        if st.button("Open Power BI prep", key="to_powerbi", use_container_width=True):
            go_to("Power BI")
    with actions[2]:
        st.caption("Downloads are generated locally on demand. Nothing is uploaded to an external service.")


def prepare_download_bundle(result: CleaningResult) -> dict[str, bytes]:
    loaded = _loaded()
    if loaded is None:
        raise RuntimeError("No active file")
    results: dict[str, CleaningResult] = st.session_state.results
    return {
        "excel": build_cleaned_workbook(results, source_name=loaded.filename),
        "csv": build_cleaned_csv(result),
        "quality": build_quality_report(result, source_name=loaded.filename),
        "log": build_activity_log_csv(result),
        "quality_csv": dataframe_to_csv(result.profile.to_frame(), neutralise=True),
    }


def download_key(result: CleaningResult) -> str:
    return f"{st.session_state.get('upload_signature')}|{st.session_state.get('clean_signature')}|{result.table_name}"


def render_downloads(result: CleaningResult, *, key_prefix: str) -> None:
    loaded = _loaded()
    if loaded is None:
        return
    key = download_key(result)
    bundle: dict[str, bytes] | None = st.session_state.get("download_bundle")
    if st.session_state.get("download_bundle_key") != key:
        bundle = None
    st.markdown("**Download cleaned outputs**")
    st.caption(
        "Excel, CSV, a multi-sheet quality report, and the audit log. "
        + ("The Excel workbook includes all cleaned tables. " if len(st.session_state.results) > 1 else "")
        + "The original upload remains available separately."
    )
    if bundle is None:
        if st.button(
            "Prepare download files",
            type="primary",
            key=f"{key_prefix}_prepare",
            use_container_width=True,
        ):
            with st.status("Generating Excel and reports locally…", expanded=True) as status:
                st.write("Formatting workbook and data-quality report…")
                try:
                    bundle = prepare_download_bundle(result)
                    st.session_state.download_bundle = bundle
                    st.session_state.download_bundle_key = key
                    status.update(label="Downloads ready", state="complete", expanded=False)
                    queue_flash("success", "Download files ready", "Excel, CSV, quality report, and audit log were generated locally.")
                    st.rerun()
                except Exception as exc:
                    status.update(label="Could not generate downloads", state="error", expanded=False)
                    st.error(safe_error_message(exc))
                    return
        st.caption("Files are generated only when you ask, so large datasets do not slow every page refresh.")
    else:
        base = safe_download_filename(loaded.filename, "")
        cols = st.columns(2)
        with cols[0]:
            st.download_button(
                "Cleaned Excel (.xlsx)",
                data=bundle["excel"],
                file_name=safe_download_filename(loaded.filename, "_cleaned.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_dl_excel",
                use_container_width=True,
                help="All cleaned tables, professional formats, filters, frozen headers, and a summary.",
            )
            st.caption(f"{format_bytes(len(bundle['excel']))} · Excel workbook")
            st.download_button(
                "Data quality report (.xlsx)",
                data=bundle["quality"],
                file_name=safe_download_filename(loaded.filename, "_data_quality.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_dl_quality",
                use_container_width=True,
                help="Profile, conversions, invalid values, duplicates, categories, outliers, and audit log.",
            )
            st.caption(f"{format_bytes(len(bundle['quality']))} · Quality report")
        with cols[1]:
            st.download_button(
                "Cleaned CSV (.csv)",
                data=bundle["csv"],
                file_name=safe_download_filename(loaded.filename, "_cleaned.csv"),
                mime="text/csv",
                key=f"{key_prefix}_dl_csv",
                use_container_width=True,
                help="UTF-8 with BOM. Formula-looking text is prefixed with an apostrophe; all other values are exported as they appear in the cleaned table.",
            )
            st.caption(f"{format_bytes(len(bundle['csv']))} · Cleaned CSV")
            st.download_button(
                "Activity log (.csv)",
                data=bundle["log"],
                file_name=safe_download_filename(loaded.filename, "_cleaning_log.csv"),
                mime="text/csv",
                key=f"{key_prefix}_dl_log",
                use_container_width=True,
            )
            st.caption(f"{format_bytes(len(bundle['log']))} · Audit log")
        with st.expander("More download options", expanded=False):
            st.download_button(
                "Column profile (.csv)",
                data=bundle["quality_csv"],
                file_name=safe_download_filename(loaded.filename, "_column_profile.csv"),
                mime="text/csv",
                key=f"{key_prefix}_dl_profile",
                use_container_width=True,
            )
            st.download_button(
                "Original upload (unchanged)",
                data=loaded.original_bytes,
                file_name=loaded.filename,
                mime="application/octet-stream",
                key=f"{key_prefix}_dl_original",
                use_container_width=True,
            )
    st.caption(
        "CSV safety: cells that begin with =, +, -, or @ are prefixed with an apostrophe to prevent spreadsheet formula injection. "
        "Excel text cells are forced to remain text."
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def _sync_active_table(loaded: LoadedFile, key: str = "active_table") -> TableData:
    names = [table.name for table in loaded.tables]
    if st.session_state.get("active_table") not in names:
        st.session_state.active_table = names[0]
    if len(names) > 1:
        st.selectbox(
            "Table",
            names,
            key=key,
            format_func=lambda name: next(
                (table.label for table in loaded.tables if table.name == name), name
            ),
        )
    return loaded.table_by_name(st.session_state.active_table) or loaded.primary_table


def _filter_profile_key(result: CleaningResult, state: FilterState, filtered: pd.DataFrame) -> str:
    payload = {
        "categories": {key: [str(value) for value in values] for key, values in state.categories.items()},
        "date": state.date_column,
        "start": str(state.start),
        "end": str(state.end),
        "rows": len(filtered),
    }
    return f"{st.session_state.get('clean_signature')}|{result.table_name}|{json.dumps(payload, sort_keys=True)}"


def render_analytics_filters(profile: DataProfile, df: pd.DataFrame) -> tuple[FilterState, pd.DataFrame]:
    token = st.session_state.get("filter_token", 0)
    categories = filterable_columns(profile)[:4]
    dates = profile.date_profiles
    state = FilterState(categories={}, date_column=None, start=None, end=None)

    with st.expander("Filter dashboard", expanded=False):
        st.caption("Filters apply to KPIs, auto charts, the manual builder, and the data table. Empty category selection means all values.")
        cat_columns = st.columns(max(1, min(3, len(categories)))) if categories else []
        for index, column_profile in enumerate(categories[:3]):
            with cat_columns[index] if cat_columns else st.container():
                try:
                    options = sorted(df[column_profile.name].dropna().astype(str).unique().tolist())
                except Exception:
                    options = []
                if len(options) > 60:
                    options = options[:60]
                selected = st.multiselect(
                    column_profile.name,
                    options,
                    key=f"filter_{token}_{column_profile.name}",
                    placeholder="All values",
                )
                if selected:
                    state.categories[column_profile.name] = selected
        if dates:
            date_profile = dates[0]
            try:
                series = pd.to_datetime(df[date_profile.name], errors="coerce").dropna()
                min_date = series.min().date() if not series.empty else date.today()
                max_date = series.max().date() if not series.empty else date.today()
            except Exception:
                min_date = max_date = date.today()
            date_cols = st.columns(2)
            with date_cols[0]:
                start = st.date_input("From date", value=min_date, min_value=min_date, max_value=max_date, key=f"filter_from_{token}")
            with date_cols[1]:
                end = st.date_input("To date", value=max_date, min_value=min_date, max_value=max_date, key=f"filter_to_{token}")
            if start > end:
                start, end = end, start
            if start != min_date or end != max_date:
                state.date_column = date_profile.name
                state.start = start
                state.end = end
        if st.button("Reset filters", key=f"reset_filters_{token}"):
            st.session_state.filter_token = token + 1
            st.session_state.filter_profiles = {}
            st.rerun()
    filtered = apply_filters(df, state)
    return state, filtered


def get_filtered_profile(result: CleaningResult, state: FilterState, filtered: pd.DataFrame) -> DataProfile:
    if not state.is_active:
        return result.profile
    cache: dict[str, DataProfile] = st.session_state.setdefault("filter_profiles", {})
    key = _filter_profile_key(result, state, filtered)
    if key not in cache:
        cache[key] = profile_dataframe(filtered, result.column_meta)
        if len(cache) > 4:
            for old_key in list(cache.keys())[:-4]:
                cache.pop(old_key, None)
    return cache[key]


def render_analytics() -> None:
    ui.page_header(
        "Analytics",
        "Explore the cleaned data",
        "KPI cards, automatic chart suggestions, a manual chart builder, filters, and export controls.",
        badges=["Plotly dashboard", "Filterable", "No Power BI required"],
    )
    loaded = _loaded()
    results: dict[str, CleaningResult] = st.session_state.get("results") or {}
    if loaded is None or not results:
        ui.empty_state(
            "AN",
            "No cleaned data yet",
            "Upload and clean a dataset first. Analytics works from the cleaned result, not the raw upload.",
        )
        if st.button("Go to Upload & Clean", type="primary", key="analytics_empty_cta"):
            go_to("Upload & Clean")
        return

    active_table = _sync_active_table(loaded)
    result = results.get(active_table.name) or _active_result()
    if result is None:
        st.warning("The cleaned result for this table is not available.")
        return
    df = result.cleaned
    if df.empty:
        ui.empty_state("0", "The cleaned table is empty", "All rows were removed as duplicates or empty rows. Review the cleaning options.")
        return

    state, filtered = render_analytics_filters(result.profile, df)
    if filtered.empty:
        st.warning("No rows match the current filters. Reset filters to see the dashboard.")
        filtered = df.iloc[0:0]
    profile = get_filtered_profile(result, state, filtered)
    summary = filtered_summary(filtered)

    if state.is_active or len(filtered) != len(df):
        ui.render_html(
            f'<div class="cf-chip-row">{ui.badge(f"Showing {len(filtered):,} of {len(df):,} rows", "cyan")}'
            + "".join(ui.badge(f"{key}: {len(values)} selected", "grey") for key, values in state.categories.items())
            + "</div>"
        )

    kpis = recommend_kpis(filtered, profile, duplicates_removed=result.summary.duplicates_removed)
    ui.kpi_grid(kpis)

    tabs = st.tabs(["Overview", "Build a chart", "Data table", "Quality & outliers", "Export"])
    with tabs[0]:
        specs = recommend_charts(filtered, profile, max_charts=6)
        if not specs:
            ui.empty_state(
                "CH",
                "No meaningful chart is supported",
                "This table may contain only text, constant, or identifier columns. Use the manual builder to try a specific field.",
            )
        else:
            charts = build_auto_charts(filtered, profile, specs)
            columns = st.columns(2, gap="large")
            for index, (spec, fig) in enumerate(charts):
                with columns[index % 2]:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"displayModeBar": False, "responsive": True},
                        key=f"auto_chart_{active_table.name}_{spec.id}",
                    )
                    ui.render_html(f'<div class="cf-chart-note">{ui.esc(spec.reason)}</div>')

    with tabs[1]:
        render_manual_chart_builder(filtered, profile, active_table.name)

    with tabs[2]:
        st.caption(f"Showing up to 500 of {len(filtered):,} filtered rows.")
        st.dataframe(filtered.head(500), width="stretch", hide_index=True, height=420)
        st.download_button(
            "Download filtered view (.csv)",
            data=dataframe_to_csv(filtered, neutralise=True),
            file_name=safe_download_filename(loaded.filename, "_filtered.csv"),
            mime="text/csv",
            key=f"analytics_filtered_csv_{active_table.name}",
        )

    with tabs[3]:
        render_quality_tab(result, profile, filtered)

    with tabs[4]:
        render_analytics_exports(result, filtered, active_table.name)


def render_manual_chart_builder(df: pd.DataFrame, profile: DataProfile, table_name: str) -> None:
    controls = st.columns([1.1, 1.2, 1.2, 1.2])
    with controls[0]:
        type_label = st.selectbox("Chart type", list(CHART_TYPES.keys()), key=f"builder_kind_{table_name}")
    kind = CHART_TYPES[type_label]

    numeric = [item.name for item in profile.numeric_profiles if item.unique > 1]
    categories = [
        item.name for item in profile.column_profiles
        if item.role in {"dimension", "date"} and item.kind != ColumnKind.IDENTIFIER
    ]
    all_columns = [str(column) for column in df.columns]
    x = y = color = None
    agg = "count"
    limit = 10

    if kind in {"bar", "line", "donut", "pie"}:
        with controls[1]:
            x_options = categories or all_columns
            x = st.selectbox("Category / date", x_options, key=f"builder_x_{table_name}")
        with controls[2]:
            y_options = ["(count records)"] + numeric
            y_choice = st.selectbox("Measure", y_options, key=f"builder_y_{table_name}")
            y = None if y_choice == "(count records)" else y_choice
        with controls[3]:
            if kind in {"bar", "line"} and y:
                agg_label = st.selectbox("Aggregation", ["Sum", "Average", "Median", "Minimum", "Maximum"], key=f"builder_agg_{table_name}")
                agg = AGGREGATIONS[agg_label]
            elif kind in {"donut", "pie"} and y:
                agg_label = st.selectbox("Aggregation", ["Sum", "Average"], key=f"builder_agg_{table_name}")
                agg = AGGREGATIONS[agg_label]
            else:
                agg = "count"
        limit = st.slider("Maximum categories", 3, 25, 10, key=f"builder_limit_{table_name}")
        if kind in {"bar", "line"}:
            color_options = ["None"] + [name for name in categories if name != x][:6]
            color_choice = st.selectbox("Color / split by", color_options, key=f"builder_color_{table_name}")
            color = None if color_choice == "None" else color_choice
    elif kind == "histogram":
        with controls[1]:
            x = st.selectbox("Numeric column", numeric or all_columns, key=f"builder_x_{table_name}")
        with controls[2]:
            color_options = ["None"] + categories[:6]
            color_choice = st.selectbox("Color by", color_options, key=f"builder_color_{table_name}")
            color = None if color_choice == "None" else color_choice
        with controls[3]:
            limit = st.slider("Bins", 8, 60, 30, key=f"builder_limit_{table_name}")
    elif kind == "scatter":
        with controls[1]:
            x = st.selectbox("X axis", numeric or all_columns, key=f"builder_x_{table_name}")
        with controls[2]:
            y = st.selectbox("Y axis", numeric or all_columns, key=f"builder_y_{table_name}")
        with controls[3]:
            color_options = ["None"] + categories[:6]
            color_choice = st.selectbox("Color by", color_options, key=f"builder_color_{table_name}")
            color = None if color_choice == "None" else color_choice
        limit = st.slider("Maximum points", 500, 5000, 2000, step=500, key=f"builder_limit_{table_name}")
    elif kind == "box":
        with controls[1]:
            x = st.selectbox("Category", categories or all_columns, key=f"builder_x_{table_name}")
        with controls[2]:
            y = st.selectbox("Numeric measure", numeric or all_columns, key=f"builder_y_{table_name}")
    else:
        with controls[1]:
            st.caption("Uses the numeric columns in the current filtered view.")

    if st.button("Build chart", type="primary", key=f"builder_build_{table_name}"):
        fig = build_chart(
            df,
            kind=kind,
            x=x,
            y=y,
            color=color,
            agg=agg,
            limit=limit,
            title=None,
        )
    else:
        fig = empty_figure("Choose your fields and click Build chart.", title="Manual chart builder")
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": True, "responsive": True},
        key=f"builder_chart_{table_name}",
    )
    st.caption("Identifier columns are intentionally excluded from category suggestions to avoid meaningless charts.")


def render_quality_tab(result: CleaningResult, profile: DataProfile, filtered: pd.DataFrame) -> None:
    outliers = result.outliers_frame()
    if outliers.empty:
        ui.callout("No outliers flagged", "Either the data has no IQR outliers or outlier flagging was disabled. Nothing is removed automatically.", tone="success")
    else:
        ui.callout(
            "Outliers are kept",
            f"{len(outliers)} example(s) fall outside 1.5× the interquartile range. Review them; CleanFlow never deletes them automatically.",
            tone="warn",
        )
        st.dataframe(outliers, width="stretch", hide_index=True)
    profile_frame = profile.to_frame()
    if not profile_frame.empty:
        st.markdown("**Column profile (filtered view)**")
        display = profile_frame[
            ["column", "type", "role", "missing", "missing_pct", "unique", "minimum", "maximum", "mean", "total", "outliers", "issues"]
        ].rename(
            columns={
                "column": "Column", "type": "Type", "role": "Role", "missing": "Missing",
                "missing_pct": "Missing %", "unique": "Unique", "minimum": "Min",
                "maximum": "Max", "mean": "Mean", "total": "Total",
                "outliers": "Outliers", "issues": "Issues",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True, height=360)
    if result.invalid_values:
        with st.expander(f"Invalid conversion values ({len(result.invalid_values)} shown)"):
            st.dataframe(pd.DataFrame(result.invalid_values), width="stretch", hide_index=True)


def render_analytics_exports(result: CleaningResult, filtered: pd.DataFrame, table_name: str) -> None:
    loaded = _loaded()
    if loaded is None:
        return
    st.markdown("**Export controls**")
    st.caption("CSV files are neutralised for spreadsheet formula safety. Excel and the quality report are generated on demand.")
    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "Cleaned CSV",
            data=build_cleaned_csv(result),
            file_name=safe_download_filename(loaded.filename, "_cleaned.csv"),
            mime="text/csv",
            key=f"analytics_clean_csv_{table_name}",
            use_container_width=True,
        )
        st.download_button(
            "Filtered CSV",
            data=dataframe_to_csv(filtered, neutralise=True),
            file_name=safe_download_filename(loaded.filename, "_filtered.csv"),
            mime="text/csv",
            key=f"analytics_filter_csv_{table_name}",
            use_container_width=True,
        )
    with cols[1]:
        st.download_button(
            "Column profile CSV",
            data=dataframe_to_csv(result.profile.to_frame(), neutralise=True),
            file_name=safe_download_filename(loaded.filename, "_column_profile.csv"),
            mime="text/csv",
            key=f"analytics_profile_csv_{table_name}",
            use_container_width=True,
        )
        if st.button("Excel, quality report & audit log", key=f"analytics_prepare_{table_name}", use_container_width=True):
            go_to("Upload & Clean")
    st.caption("The complete Excel bundle is available under Upload & Clean → Downloads.")


# ---------------------------------------------------------------------------
# Power BI
# ---------------------------------------------------------------------------
def get_powerbi_model() -> Any:
    results = st.session_state.get("results") or {}
    key = f"{st.session_state.get('upload_signature')}|{st.session_state.get('clean_signature')}|{'|'.join(results.keys())}"
    if st.session_state.get("powerbi_key") != key or st.session_state.get("powerbi_model") is None:
        with st.spinner("Preparing the suggested Power BI model…"):
            model = prepare_model(results)
        st.session_state.powerbi_key = key
        st.session_state.powerbi_model = model
        st.session_state.powerbi_package = None
        st.session_state.powerbi_package_key = None
    return st.session_state.powerbi_model


def render_powerbi() -> None:
    ui.page_header(
        "Power BI analytics",
        "A Power BI-ready dataset, not a .pbix file",
        "Prepare a clean star-schema dataset, relationship guidance, DAX measures, a data dictionary, "
        "and import instructions. You build the report in Power BI Desktop.",
        badges=["Star-schema prep", "DAX suggestions", "No .pbix claim"],
    )
    loaded = _loaded()
    results: dict[str, CleaningResult] = st.session_state.get("results") or {}
    if loaded is None or not results:
        ui.empty_state(
            "BI",
            "Clean a dataset first",
            "Power BI preparation uses the cleaned output, so the model and measures reflect the reviewed data.",
        )
        if st.button("Go to Upload & Clean", type="primary", key="powerbi_empty_cta"):
            go_to("Upload & Clean")
        return

    ui.feature_grid(
        [
            ("WEB", "Dashboard preview (this website)", "Interactive Plotly charts inside CleanFlow. Not a Power BI report."),
            ("DATA", "Power BI-ready dataset (download here)", "CSV and Excel tables plus a suggested star schema, relationships, and DAX."),
            ("PBIX", "Power BI report (.pbix)", "Created and saved by you in Power BI Desktop. This application does not generate .pbix files."),
        ]
    )
    ui.callout(
        "Know what you are downloading",
        "The package is a data/model blueprint. It does not contain a report file. Power BI Desktop is free for local report authoring. "
        "CleanFlow does not upload your data to Microsoft.",
        tone="info",
    )

    try:
        model = get_powerbi_model()
    except Exception as exc:
        st.error(safe_error_message(exc))
        return

    ui.kpi_grid(
        [
            {"label": "Model tables", "display": format_number(len(model.tables)), "hint": "Fact, dimension, and flat tables"},
            {"label": "Fact tables", "display": format_number(len(model.fact_tables)), "hint": "Tables with measures"},
            {"label": "Dimensions", "display": format_number(len(model.dimension_tables)), "hint": "Surrogate-keyed lookups"},
            {"label": "Relationships", "display": format_number(len(model.relationships)), "hint": "Suggested Many-to-One links"},
            {"label": "DAX measures", "display": format_number(len(model.measures)), "hint": "Definitions you can paste"},
            {"label": "Source tables", "display": format_number(len(model.source_tables)), "hint": "Cleaned tables included"},
        ]
    )

    tabs = st.tabs(["Model", "DAX & KPIs", "Data dictionary", "Import guide", "Download package"])
    with tabs[0]:
        st.markdown("**Suggested model**")
        st.dataframe(model.model_frame, width="stretch", hide_index=True)
        st.markdown("**Suggested relationships**")
        if model.relationships:
            st.dataframe(model.relationships_frame, width="stretch", hide_index=True)
            st.caption("Many-to-One from fact to dimension, single cross-filter direction. Confirm keys and uniqueness before use.")
        else:
            st.info("No relationships were suggested. A flat table is the safer model for this data.")
        if model.notes:
            with st.expander("Model notes and assumptions", expanded=False):
                for note in model.notes:
                    st.markdown(f"- {note}")
        st.markdown("**Table roles**")
        role_rows = []
        for table in model.tables:
            role_rows.append(
                {
                    "Table": table.name,
                    "Role": table.role,
                    "Rows": f"{len(table.dataframe):,}",
                    "Columns": table.dataframe.shape[1],
                    "Purpose": table.description,
                }
            )
        st.dataframe(pd.DataFrame(role_rows), width="stretch", hide_index=True)

    with tabs[1]:
        if model.measures:
            names = [measure.name for measure in model.measures]
            selected = st.selectbox("Inspect a DAX measure", names, key="dax_measure_select")
            measure = next((item for item in model.measures if item.name == selected), model.measures[0])
            st.markdown(f"**{measure.name}** · {measure.description}")
            st.code(f"{measure.name} = {measure.expression}", language="dax")
            st.caption(f"Suggested format string: `{measure.format_string}`")
            st.dataframe(model.measures_frame, width="stretch", hide_index=True, height=300)
            with st.expander("All DAX measures (plain text)", expanded=False):
                st.code(model.dax_text(), language="dax")
        else:
            st.info("No numeric measures were found, so no DAX measures were suggested.")
        st.markdown("**KPI suggestions**")
        if not model.kpi_recommendations.empty:
            st.dataframe(model.kpi_recommendations, width="stretch", hide_index=True)
        st.markdown("**Visual ideas**")
        if not model.visuals_frame.empty:
            st.dataframe(model.visuals_frame, width="stretch", hide_index=True)
        st.caption("Time-intelligence measures are omitted until a date table is marked inside Power BI.")

    with tabs[2]:
        st.dataframe(model.data_dictionary, width="stretch", hide_index=True, height=420)
        st.download_button(
            "Download data dictionary (.csv)",
            data=dataframe_to_csv(model.data_dictionary, neutralise=False),
            file_name="cleanflow_powerbi_data_dictionary.csv",
            mime="text/csv",
            key="pbi_dictionary_csv",
        )
        st.caption("Dates, measures, keys, and dimension attributes are described from the cleaned profile.")

    with tabs[3]:
        guide = _powerbi_guide_text(model)
        st.markdown(guide)
        st.download_button(
            "Download import guide (.md)",
            data=guide.encode("utf-8"),
            file_name="cleanflow_powerbi_import_guide.md",
            mime="text/markdown",
            key="pbi_guide_md",
        )
        st.caption("Power BI Desktop is a separate free Windows application from Microsoft. CleanFlow does not automate it.")

    with tabs[4]:
        st.markdown("**Build the package**")
        st.caption(
            "The ZIP contains CSVs, a documentation workbook, data dictionary, relationships, DAX, model.json, and the import guide. "
            "No .pbix is included."
        )
        key = st.session_state.get("powerbi_key")
        package = st.session_state.get("powerbi_package")
        if st.session_state.get("powerbi_package_key") != key:
            package = None
        if package is None:
            if st.button("Generate Power BI package", type="primary", key="pbi_generate", use_container_width=True):
                with st.status("Building the Power BI package locally…", expanded=True) as status:
                    st.write("Writing CSVs, workbook, model documentation, and DAX…")
                    try:
                        package = build_powerbi_package(results)
                        st.session_state.powerbi_package = package
                        st.session_state.powerbi_package_key = key
                        status.update(label="Package ready", state="complete", expanded=False)
                        queue_flash("success", "Power BI package ready", "The ZIP was generated locally and contains no .pbix file.")
                        st.rerun()
                    except Exception as exc:
                        status.update(label="Package generation failed", state="error", expanded=False)
                        st.error(safe_error_message(exc))
                        return
            st.caption("Large tables can take a few seconds. Files are created only when requested.")
        else:
            st.download_button(
                "Download Power BI package (.zip)",
                data=package.zip_bytes,
                file_name=safe_download_filename(loaded.filename, "_powerbi_package.zip"),
                mime="application/zip",
                key="pbi_download_zip",
                type="primary",
                use_container_width=True,
            )
            st.caption(f"{format_bytes(len(package.zip_bytes))} · ZIP archive")
            cols = st.columns(2)
            with cols[0]:
                st.download_button(
                    "Data dictionary CSV",
                    data=package.dictionary_csv,
                    file_name="cleanflow_data_dictionary.csv",
                    mime="text/csv",
                    key="pbi_pkg_dictionary",
                    use_container_width=True,
                )
                st.download_button(
                    "DAX measures (.dax)",
                    data=package.dax_text.encode("utf-8"),
                    file_name="cleanflow_dax_measures.dax",
                    mime="text/plain",
                    key="pbi_pkg_dax",
                    use_container_width=True,
                )
            with cols[1]:
                st.download_button(
                    "Relationships CSV",
                    data=package.relationships_csv,
                    file_name="cleanflow_relationships.csv",
                    mime="text/csv",
                    key="pbi_pkg_relationships",
                    use_container_width=True,
                )
                st.download_button(
                    "Import guide (.md)",
                    data=package.import_guide.encode("utf-8"),
                    file_name="cleanflow_import_guide.md",
                    mime="text/markdown",
                    key="pbi_pkg_guide",
                    use_container_width=True,
                )
            with st.expander("Package contents", expanded=True):
                st.markdown(
                    "- `tables/*.csv` — one CSV per model table (UTF-8 with BOM)\n"
                    "- `CleanFlow_PowerBI_Model.xlsx` — tables plus model documentation\n"
                    "- `model.json` — model metadata for tooling\n"
                    "- `relationships.csv`, `data_dictionary.csv`, `kpi_recommendations.csv`, `visual_recommendations.csv`\n"
                    "- `dax_measures.dax` — paste-ready measure definitions\n"
                    "- `README_IMPORT.md`, `SECURITY_AND_LIMITATIONS.txt`\n"
                )
            ui.callout(
                "CSV import safety",
                "The package CSVs preserve raw values for Power BI import and are not neutralised for Excel formula execution. "
                "Import them into Power BI; do not double-click them in Excel.",
                tone="warn",
            )


def _powerbi_guide_text(model: Any) -> str:
    relationship_lines = "\n".join(
        f"| {relationship.from_table} | {relationship.from_column} | {relationship.to_table} | {relationship.to_column} | {relationship.cardinality} |"
        for relationship in model.relationships
    ) or "| — | — | — | — | No relationships suggested |"
    return f"""### Import this package into Power BI Desktop

1. Install **Power BI Desktop** (free, Windows) from Microsoft.
2. Unzip the downloaded package and open Power BI Desktop.
3. **Get data → Text/CSV**, then import each file from `tables/`.
   Choose **File origin: 65001: Unicode (UTF-8)** and comma delimiter.
4. In Power Query, check types: dates as Date, measures as Decimal Number or Whole Number,
   and identifiers as Text. Do not turn IDs into numbers.
5. In **Model view**, create these suggested relationships (fact → dimension):

| From table | From column | To table | To column | Cardinality |
|---|---|---|---|---|
{relationship_lines}

6. Set each relationship to **Many to one (*:1)** and **Single** cross-filter direction.
7. If a `Dim_Date` table exists, select it and use **Table tools → Mark as date table**,
   choosing its `date` column.
8. Create the measures from `dax_measures.dax` using **Modeling → New measure**.
   Apply the suggested format strings.
9. Build visuals using `visual_recommendations.csv`.
10. Save your own `.pbix` file. CleanFlow does not produce it.

### Refresh workflow

Download a new package when the source changes, replace the CSV files (keeping names),
and press **Refresh** in Power BI Desktop. Relationships and measures in your saved report
are preserved.

### Limitations to review

- Relationship suggestions are inferred from naming, uniqueness, and cardinality — not
  guaranteed business rules. Confirm them.
- Mixed currencies are not summed in suggested measures.
- Percentages are stored as percentage points and formatted with `0.0"%"`.
- A star schema is created only when a usable dimension/date and a numeric measure exist;
  otherwise the cleaned table is exported flat.
"""


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def render_history() -> None:
    ui.page_header(
        "Cleaning history",
        "Processing metadata, not your data",
        "Timestamps, filenames, counts, operations, and status are stored locally. "
        "Cell values and uploaded files are never written to history.",
        badges=["Metadata only", "Local JSON", "50 job cap"],
    )
    jobs = load_jobs()
    stats = history_stats(jobs)
    ui.kpi_grid(
        [
            {"label": "Total jobs", "display": format_number(stats["total_jobs"]), "hint": "Completed and failed"},
            {"label": "Completed", "display": format_number(stats["completed_jobs"]), "hint": "Cleaning runs"},
            {"label": "Rows cleaned", "display": format_number(stats["total_rows_cleaned"]), "hint": "Output rows"},
            {"label": "Duplicates removed", "display": format_number(stats["duplicates_removed"]), "hint": "Across completed jobs"},
            {"label": "Files seen", "display": format_number(stats["files"]), "hint": "Distinct filenames"},
        ]
    )
    if not jobs:
        ui.empty_state(
            "HI",
            "No history yet",
            "Complete a cleaning run to create a metadata-only history entry. Failed loads are recorded too, without file content.",
        )
        return

    filters = st.columns([2, 1, 1])
    with filters[0]:
        search = st.text_input("Search filename", placeholder="Type part of a filename…", key="history_search")
    with filters[1]:
        status_choice = st.selectbox("Status", ["All", "Completed", "Failed"], key="history_status")
    with filters[2]:
        ui.render_html("<div style='height:27px'></div>")
        st.download_button(
            "Export history CSV",
            data=export_history_csv(),
            file_name="cleanflow_history.csv",
            mime="text/csv",
            key="history_download",
            use_container_width=True,
        )

    filtered_jobs = jobs
    if search.strip():
        needle = search.strip().lower()
        filtered_jobs = [job for job in filtered_jobs if needle in str(job.get("filename", "")).lower()]
    if status_choice != "All":
        filtered_jobs = [job for job in filtered_jobs if str(job.get("status", "")).lower() == status_choice.lower()]

    if not filtered_jobs:
        st.info("No jobs match the current search and status filters.")
    else:
        st.dataframe(history_frame(filtered_jobs), width="stretch", hide_index=True, height=320)

        options = {
            f"{ui.format_timestamp(job.get('timestamp'))} · {job.get('filename', '')} · {job.get('status', '').title()}": job
            for job in filtered_jobs
        }
        selected_label = st.selectbox("Inspect a job", list(options.keys()), key="history_selected")
        selected = options[selected_label]
        tone = "green" if selected.get("status") == "completed" else "red"
        duration_label = f"{int(selected.get('duration_ms') or 0) / 1000:.1f}s"
        ui.render_html(
            f'<div class="cf-chip-row">{ui.badge(str(selected.get("status", "")).title(), tone)}'
            f'{ui.badge(str(selected.get("file_type", "")).upper(), "grey")}'
            f'{ui.badge(duration_label, "grey")}</div>'
        )
        detail = pd.DataFrame(
            [
                {"Field": "File", "Value": selected.get("filename", "")},
                {"Field": "Tables", "Value": ", ".join(selected.get("tables", [])) or "—"},
                {"Field": "Rows before / after", "Value": f"{selected.get('rows_before', 0):,} → {selected.get('rows_after', 0):,}"},
                {"Field": "Duplicates removed", "Value": f"{selected.get('duplicates_removed', 0):,}"},
                {"Field": "Missing cells before / after", "Value": f"{selected.get('missing_before', 0):,} → {selected.get('missing_after', 0):,}"},
                {"Field": "Operations", "Value": ", ".join(selected.get("operations", [])) or "—"},
                {"Field": "Outputs available", "Value": ", ".join(selected.get("outputs", [])) or "—"},
            ]
        )
        st.dataframe(detail, width="stretch", hide_index=True)
        if selected.get("error"):
            ui.callout("Failure reason", selected["error"], tone="danger")
        st.caption("History entries contain no cell values, no file bytes, and no download contents.")

    with st.expander("Clear cleaning history", expanded=False):
        st.warning("This removes all metadata entries. Generated download files that you already saved are not affected.")
        confirm = st.checkbox("I understand that this cannot be undone.", key="history_clear_confirm")
        if st.button("Clear all history", key="history_clear_button", disabled=not confirm):
            clear_jobs()
            queue_flash("success", "History cleared", "All metadata-only entries were removed.")
            st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Home": render_home,
    "Upload & Clean": render_upload_clean,
    "Analytics": render_analytics,
    "Power BI": render_powerbi,
    "History": render_history,
}


def main() -> None:
    init_state()
    ui.inject_styles()
    render_sidebar()
    render_flashes()
    page = st.session_state.get("page", "Home")
    if page not in PAGES:
        page = "Home"
    try:
        PAGES[page]()
    except Exception as exc:  # pragma: no cover - user-facing safety boundary
        LOGGER.exception("Unhandled page error")
        st.error(safe_error_message(exc))
        st.caption(
            "The application stayed running. Try clearing the current dataset or reloading the page. "
            "No uploaded file content is included in this message."
        )


if __name__ == "__main__":
    main()
