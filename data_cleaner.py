"""Configurable, auditable data cleaning engine.

Design rules:

* Cleaning never invents values: missing cells stay missing.
* Destructive operations (duplicate removal, empty row/column removal) are
  explicit options, are counted, and are written to the audit log.
* Exact duplicates are determined using **all** columns, never one matching ID.
* Identifier columns (IDs, phone numbers, postal codes) are kept as text so
  leading zeros survive.
* When too many values in a column fail type conversion, the column is kept as
  text rather than dropping the values that did not parse.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from modules.data_profiler import (
    ColumnKind,
    ColumnProfile,
    ConversionSettings,
    DataProfile,
    analyse_and_convert,
    looks_like_identifier_name,
    profile_dataframe,
)

LOGGER = logging.getLogger("cleanflow.cleaner")

DEFAULT_MISSING_TOKENS: tuple[str, ...] = (
    "",
    "n/a",
    "n.a.",
    "null",
    "none",
    "nan",
    "nil",
    "not available",
)

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"}
_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
}
_NON_BREAKING = {"\u00a0": " ", "\u202f": " ", "\u2007": " "}


@dataclass
class CleaningOptions:
    """All cleaning operations can be switched on/off from the UI."""

    standardize_columns: bool = True
    trim_whitespace: bool = True
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    normalize_missing: bool = True
    missing_tokens: tuple[str, ...] = DEFAULT_MISSING_TOKENS
    normalize_category_case: bool = False
    parse_types: bool = True
    convert_currency: bool = True
    convert_percentage: bool = True
    parse_dates: bool = True
    day_first: str = "auto"  # auto | day_first | month_first
    decimal_style: str = "auto"  # auto | dot | comma
    numeric_threshold: float = 0.95
    date_threshold: float = 0.95
    remove_duplicates: bool = True
    drop_empty_rows: bool = True
    drop_empty_columns: bool = True
    detect_outliers: bool = True

    def __post_init__(self) -> None:
        if self.day_first not in {"auto", "day_first", "month_first"}:
            self.day_first = "auto"
        if self.decimal_style not in {"auto", "dot", "comma"}:
            self.decimal_style = "auto"
        self.numeric_threshold = min(max(float(self.numeric_threshold), 0.5), 1.0)
        self.date_threshold = min(max(float(self.date_threshold), 0.5), 1.0)
        if isinstance(self.missing_tokens, list):
            self.missing_tokens = tuple(self.missing_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "Standardize column names": self.standardize_columns,
            "Trim whitespace": self.trim_whitespace,
            "Normalize text formatting": self.normalize_unicode and self.collapse_whitespace,
            "Normalize missing-value tokens": self.normalize_missing,
            "Missing tokens treated as blank": ", ".join(
                repr(token) if token == "" else token for token in self.missing_tokens
            ),
            "Normalize category letter case": self.normalize_category_case,
            "Detect and convert types": self.parse_types,
            "Convert currency to numbers": self.convert_currency,
            "Convert percentages to points": self.convert_percentage,
            "Parse dates": self.parse_dates,
            "Date order": {
                "auto": "Auto (day-first when ambiguous)",
                "day_first": "Day first (DD/MM/YYYY)",
                "month_first": "Month first (MM/DD/YYYY)",
            }.get(self.day_first, self.day_first),
            "Decimal separator": {
                "auto": "Auto-detect",
                "dot": "Dot decimal (1,234.56)",
                "comma": "Comma decimal (1.234,56)",
            }.get(self.decimal_style, self.decimal_style),
            "Numeric validity threshold": f"{self.numeric_threshold:.0%}",
            "Date validity threshold": f"{self.date_threshold:.0%}",
            "Remove exact duplicate rows": self.remove_duplicates,
            "Remove empty rows": self.drop_empty_rows,
            "Remove empty columns": self.drop_empty_columns,
            "Flag outliers (never remove)": self.detect_outliers,
        }

    def fingerprint(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


@dataclass
class AuditEntry:
    step: int
    action: str
    description: str
    columns: list[str] = field(default_factory=list)
    rows_affected: int = 0
    cells_affected: int = 0
    status: str = "completed"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "action": self.action,
            "description": self.description,
            "columns": ", ".join(self.columns[:25]) + ("…" if len(self.columns) > 25 else ""),
            "rows_affected": self.rows_affected,
            "cells_affected": self.cells_affected,
            "status": self.status,
        }


@dataclass
class CleaningSummary:
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    missing_cells_before: int = 0
    missing_cells_after: int = 0
    duplicates_removed: int = 0
    empty_rows_found: int = 0
    empty_rows_removed: int = 0
    empty_columns_found: int = 0
    empty_columns_removed: int = 0
    missing_tokens_replaced: int = 0
    cells_trimmed: int = 0
    cells_text_normalized: int = 0
    categories_normalized: int = 0
    columns_renamed: int = 0
    values_converted: int = 0
    invalid_values: int = 0
    outliers_flagged: int = 0
    category_groups_detected: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "missing_cells_before": self.missing_cells_before,
            "missing_cells_after": self.missing_cells_after,
            "duplicates_removed": self.duplicates_removed,
            "empty_rows_found": self.empty_rows_found,
            "empty_rows_removed": self.empty_rows_removed,
            "empty_columns_found": self.empty_columns_found,
            "empty_columns_removed": self.empty_columns_removed,
            "missing_tokens_replaced": self.missing_tokens_replaced,
            "cells_trimmed": self.cells_trimmed,
            "cells_text_normalized": self.cells_text_normalized,
            "categories_normalized": self.categories_normalized,
            "columns_renamed": self.columns_renamed,
            "values_converted": self.values_converted,
            "invalid_values": self.invalid_values,
            "outliers_flagged": self.outliers_flagged,
            "category_groups_detected": self.category_groups_detected,
            "duration_ms": self.duration_ms,
        }


@dataclass
class CleaningResult:
    table_name: str
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    options: CleaningOptions
    summary: CleaningSummary
    audit: list[AuditEntry]
    profile: DataProfile
    column_meta: dict[str, dict[str, Any]]
    conversions: list[dict[str, Any]] = field(default_factory=list)
    invalid_values: list[dict[str, Any]] = field(default_factory=list)
    category_groups: list[dict[str, Any]] = field(default_factory=list)
    duplicates_removed_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    column_renames: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def audit_frame(self) -> pd.DataFrame:
        if not self.audit:
            return pd.DataFrame(columns=["timestamp", "step", "action", "description", "columns", "rows_affected", "cells_affected"])
        return pd.DataFrame([entry.to_dict() for entry in self.audit])

    def quality_frame(self) -> pd.DataFrame:
        return self.profile.to_frame()

    def issues_frame(self) -> pd.DataFrame:
        rows = [
            {"column": profile.name, "type": profile.kind_label, "issues": issue}
            for profile in self.profile.column_profiles
            for issue in profile.issues
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column", "type", "issues"])

    def outliers_frame(self) -> pd.DataFrame:
        rows = []
        for profile in self.profile.column_profiles:
            for example in profile.outlier_examples:
                rows.append(
                    {
                        "column": profile.name,
                        "row": example.get("row"),
                        "value": example.get("value"),
                        "note": "Possible outlier (IQR rule); not removed",
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column", "row", "value", "note"])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    options: CleaningOptions | None = None,
    *,
    table_name: str = "Data",
) -> CleaningResult:
    """Run the full cleaning pipeline on a copy of ``df``."""

    started = time.perf_counter()
    options = options or CleaningOptions()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataframe expects a pandas DataFrame")

    raw = df.copy().reset_index(drop=True)
    work = df.copy().reset_index(drop=True)
    # Pandas returns a DataFrame instead of a Series for ``frame[column]``
    # when headers are duplicated. Give the working and preview copies stable
    # labels before any cleaning step; the caller's uploaded DataFrame is not
    # changed.
    source_columns = [str(column).strip() or f"column_{index + 1}" for index, column in enumerate(df.columns)]
    unique_columns = _unique_column_names(source_columns)
    if unique_columns != source_columns:
        raw.columns = unique_columns
        work.columns = unique_columns
    summary = CleaningSummary(
        rows_before=int(raw.shape[0]),
        columns_before=int(raw.shape[1]),
        missing_cells_before=_count_missing(raw),
    )
    audit: list[AuditEntry] = []
    warnings: list[str] = [
        "Missing values are left empty. CleanFlow does not impute or invent data.",
        "The original uploaded file is preserved separately from every cleaned output.",
    ]
    if unique_columns != source_columns:
        warnings.append("Duplicate or blank source column names were made unique in the cleaned output.")
    column_meta: dict[str, dict[str, Any]] = {}
    conversions: list[dict[str, Any]] = []
    invalid_values: list[dict[str, Any]] = []
    category_groups: list[dict[str, Any]] = []
    duplicates_frame = pd.DataFrame()
    column_renames: list[dict[str, str]] = []

    step = 0

    # -- 1. Missing-value tokens -------------------------------------------
    if options.normalize_missing:
        step += 1
        tokens = {str(token).strip().casefold() for token in options.missing_tokens}
        replaced = 0
        columns_touched: set[str] = set()
        for column in work.columns:
            if work[column].dtype != object:
                # Keep native NA in numeric/date columns; only text gets token
                # normalisation.
                continue
            new_values = []
            changed = False
            for value in work[column].tolist():
                if isinstance(value, str) and value.strip().casefold() in tokens:
                    new_values.append(None)
                    replaced += 1
                    changed = True
                else:
                    new_values.append(value)
            if changed:
                work[column] = pd.Series(new_values, index=work.index, dtype=object)
                columns_touched.add(str(column))
        summary.missing_tokens_replaced = replaced
        if replaced:
            audit.append(
                AuditEntry(
                    step=step,
                    action="normalize_missing_values",
                    description=(
                        "Replaced common blank placeholders with a true empty value. "
                        "No values were filled in."
                    ),
                    columns=sorted(columns_touched),
                    cells_affected=replaced,
                )
            )

    # -- 2. Text normalisation and trimming --------------------------------
    if options.trim_whitespace or options.normalize_unicode or options.collapse_whitespace:
        step += 1
        trimmed = 0
        normalized = 0
        columns_touched = set()
        for column in work.columns:
            if work[column].dtype != object:
                continue
            original_values = work[column].tolist()
            new_values = []
            changed = False
            for value in original_values:
                if not isinstance(value, str):
                    new_values.append(value)
                    continue
                cleaned = value
                if options.normalize_unicode:
                    cleaned = unicodedata.normalize("NFKC", cleaned)
                    cleaned = "".join(
                        _DASHES.get(char) or _QUOTES.get(char) or _NON_BREAKING.get(char) or char
                        for char in cleaned
                    )
                    cleaned = _CONTROL_RE.sub("", cleaned)
                if options.collapse_whitespace:
                    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
                if options.trim_whitespace:
                    cleaned = cleaned.strip()
                if cleaned != value:
                    changed = True
                    if options.trim_whitespace and value.strip() != value:
                        trimmed += 1
                    else:
                        normalized += 1
                new_values.append(cleaned)
            if changed:
                work[column] = pd.Series(new_values, index=work.index, dtype=object)
                columns_touched.add(str(column))
        summary.cells_trimmed = trimmed
        summary.cells_text_normalized = normalized
        if trimmed or normalized:
            audit.append(
                AuditEntry(
                    step=step,
                    action="normalize_text",
                    description=(
                        "Trimmed surrounding whitespace and normalised common text formatting "
                        "(Unicode, curly quotes, non-breaking spaces, repeated whitespace)."
                    ),
                    columns=sorted(columns_touched),
                    cells_affected=trimmed + normalized,
                )
            )

    # -- 3. Category consistency detection / optional case normalisation --
    step += 1
    category_groups, categorical_columns = _detect_category_groups(work)
    summary.category_groups_detected = sum(len(item["groups"]) for item in category_groups)
    if category_groups:
        audit.append(
            AuditEntry(
                step=step,
                action="detect_inconsistent_categories",
                description=(
                    "Detected category labels that differ only by letter case. "
                    + (
                        "They will be merged into the most common spelling."
                        if options.normalize_category_case
                        else "No labels were changed; enable category normalisation on the cleaning screen to merge them."
                    )
                ),
                columns=[item["column"] for item in category_groups],
                cells_affected=sum(
                    group["total"] - group["canonical_count"]
                    for item in category_groups
                    for group in item["groups"]
                ),
                status="completed" if options.normalize_category_case else "requires review",
            )
        )

    if options.normalize_category_case and categorical_columns:
        step += 1
        changed_total = 0
        touched: list[str] = []
        for column in categorical_columns:
            series = work[column]
            values = series.tolist()
            counts: dict[str, dict[str, Any]] = {}
            for value in values:
                if not isinstance(value, str) or not value:
                    continue
                key = value.casefold()
                entry = counts.setdefault(key, {"counts": {}, "first": value})
                entry["counts"][value] = entry["counts"].get(value, 0) + 1
            canonical: dict[str, str] = {}
            for key, entry in counts.items():
                variants = entry["counts"]
                canonical[key] = sorted(variants.items(), key=lambda item: (-item[1], item[0]))[0][0]
            new_values = []
            changed = False
            for value in values:
                if isinstance(value, str) and value:
                    replacement = canonical.get(value.casefold(), value)
                    if replacement != value:
                        changed = True
                        changed_total += 1
                    new_values.append(replacement)
                else:
                    new_values.append(value)
            if changed:
                work[column] = pd.Series(new_values, index=work.index, dtype=object)
                touched.append(str(column))
        summary.categories_normalized = changed_total
        if changed_total:
            audit.append(
                AuditEntry(
                    step=step,
                    action="normalize_category_case",
                    description=(
                        "Merged category labels that differed only by letter case, using the most "
                        "frequent spelling as the canonical label."
                    ),
                    columns=touched,
                    cells_affected=changed_total,
                )
            )

    # -- 4. Standardise column names --------------------------------------
    if options.standardize_columns:
        step += 1
        rename_map: dict[str, str] = {}
        used: set[str] = set()
        for index, column in enumerate(work.columns):
            base = standardize_column_name(column)
            candidate = base
            suffix = 2
            while candidate in used:
                candidate = f"{base}_{suffix}"
                suffix += 1
            used.add(candidate)
            if str(column) != candidate:
                rename_map[str(column)] = candidate
                column_renames.append({"before": str(column), "after": candidate})
        if rename_map:
            work = work.rename(columns=rename_map)
            summary.columns_renamed = len(rename_map)
            audit.append(
                AuditEntry(
                    step=step,
                    action="standardize_column_names",
                    description=(
                        "Converted column names to lower-case snake_case and made duplicates unique."
                    ),
                    columns=list(rename_map.values()),
                    cells_affected=len(rename_map),
                )
            )
    else:
        # Even without standardisation, exports and profiles need unique string
        # column names. Keep display names but trim surrounding whitespace.
        work.columns = _unique_column_names([str(column).strip() or f"column_{i + 1}" for i, column in enumerate(work.columns)])
        raw.columns = _unique_column_names([str(column).strip() or f"column_{i + 1}" for i, column in enumerate(raw.columns)])

    # -- 5. Type conversion ------------------------------------------------
    if options.parse_types:
        step += 1
        settings = ConversionSettings(
            parse_dates=options.parse_dates,
            convert_currency=options.convert_currency,
            convert_percentage=options.convert_percentage,
            numeric_threshold=options.numeric_threshold,
            date_threshold=options.date_threshold,
            day_first=options.day_first,
            decimal_style=options.decimal_style,
        )
        converted_columns: list[str] = []
        for column in work.columns:
            series = work[column]
            try:
                outcome = analyse_and_convert(series, str(column), settings)
            except Exception:
                LOGGER.exception("Conversion failed for column %s", column)
                warnings.append(f"Column '{column}' could not be type-checked and was kept unchanged.")
                continue

            if outcome.converted is not None:
                work[column] = outcome.converted
                converted_columns.append(str(column))
            meta = {
                "kind": outcome.kind,
                "issues": outcome.issues,
                **outcome.metadata,
            }
            column_meta[str(column)] = meta

            if outcome.converted is not None:
                old_kind = _series_kind_label(series)
                new_kind = _kind_label(outcome.kind)
                # Count failures against the original non-missing values, not
                # just the first ten examples retained for the report.
                original_non_missing = int(series.map(lambda value: not _is_blank(value)).sum())
                invalid_count = max(original_non_missing - outcome.valid_count, 0) if outcome.kind in {
                    ColumnKind.NUMERIC,
                    ColumnKind.CURRENCY,
                    ColumnKind.PERCENTAGE,
                    ColumnKind.DATE,
                    ColumnKind.DATETIME,
                } else 0
                conversion = {
                    "column": str(column),
                    "from": old_kind,
                    "to": new_kind,
                    "values_converted": outcome.valid_count,
                    "values_flagged": invalid_count,
                    "notes": " | ".join(outcome.issues),
                    "examples": "; ".join(
                        f"row {item['row']}: {item['value']}" for item in outcome.invalid_examples[:5]
                    ),
                }
                conversions.append(conversion)
                for example in outcome.invalid_examples:
                    invalid_values.append(
                        {
                            "column": str(column),
                            "row": example["row"],
                            "value": example["value"],
                            "problem": example.get("problem", "invalid value"),
                        }
                    )
                summary.values_converted += outcome.valid_count
                summary.invalid_values += invalid_count
            elif outcome.issues:
                # Text/mixed columns with conversion warnings still need a hint
                # so profiling does not misclassify them as measures.
                column_meta[str(column)]["kind"] = outcome.kind

        if conversions:
            audit.append(
                AuditEntry(
                    step=step,
                    action="convert_types",
                    description=(
                        "Detected and converted numbers, currency, percentages, dates, booleans, "
                        "and identifier columns. Columns that failed the validity threshold were kept as text."
                    ),
                    columns=[item["column"] for item in conversions],
                    cells_affected=sum(item["values_converted"] for item in conversions),
                    status="requires review" if summary.invalid_values else "completed",
                )
            )

    # -- 6. Empty rows / columns ------------------------------------------
    step += 1
    empty_row_mask = work.isna().all(axis=1) if work.shape[1] else pd.Series(False, index=work.index)
    summary.empty_rows_found = int(empty_row_mask.sum())
    if options.drop_empty_rows and summary.empty_rows_found:
        work = work.loc[~empty_row_mask].reset_index(drop=True)
        summary.empty_rows_removed = summary.empty_rows_found
        audit.append(
            AuditEntry(
                step=step,
                action="remove_empty_rows",
                description="Removed rows where every column was empty. Counts are recorded here.",
                rows_affected=summary.empty_rows_removed,
            )
        )

    if work.shape[1]:
        empty_col_mask = work.isna().all(axis=0)
        empty_columns = [str(column) for column, is_empty in empty_col_mask.items() if is_empty]
    else:
        empty_columns = []
    summary.empty_columns_found = len(empty_columns)
    if options.drop_empty_columns and empty_columns:
        work = work.drop(columns=empty_columns)
        summary.empty_columns_removed = len(empty_columns)
        for column in empty_columns:
            column_meta.pop(column, None)
        audit.append(
            AuditEntry(
                step=step,
                action="remove_empty_columns",
                description="Removed columns where every value was empty. Column names are recorded.",
                columns=empty_columns,
            )
        )

    if work.shape[1] == 0 and work.shape[0] > 0:
        warnings.append("Every column was empty. The cleaned table has no columns to analyse.")

    # -- 7. Duplicate rows -------------------------------------------------
    step += 1
    if options.remove_duplicates and work.shape[0] > 0 and work.shape[1] > 0:
        try:
            duplicate_mask = work.duplicated(keep="first")
        except TypeError:
            duplicate_mask = work.astype(str).duplicated(keep="first")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            duplicates_frame = work.loc[duplicate_mask].copy().reset_index(drop=True)
            if len(duplicates_frame) > 2000:
                duplicates_frame = duplicates_frame.head(2000)
            work = work.loc[~duplicate_mask].reset_index(drop=True)
            summary.duplicates_removed = duplicate_count
            audit.append(
                AuditEntry(
                    step=step,
                    action="remove_exact_duplicates",
                    description=(
                        "Removed rows that exactly duplicated an earlier row across ALL columns. "
                        "A single matching column is never enough to treat two records as duplicates."
                    ),
                    rows_affected=duplicate_count,
                )
            )

    work = work.reset_index(drop=True)

    # -- 8. Profile and outliers ------------------------------------------
    profile = profile_dataframe(work, column_meta)
    if options.detect_outliers:
        summary.outliers_flagged = sum(item.outlier_count for item in profile.column_profiles)
    else:
        for column_profile in profile.column_profiles:
            column_profile.outlier_count = 0
            column_profile.outlier_examples = []
            column_profile.issues = [issue for issue in column_profile.issues if "outlier" not in issue.lower()]
        summary.outliers_flagged = 0

    summary.rows_after = int(work.shape[0])
    summary.columns_after = int(work.shape[1])
    summary.missing_cells_after = int(profile.missing_cells)
    summary.duration_ms = int((time.perf_counter() - started) * 1000)

    # General warnings, kept short enough for a report.
    if summary.duplicates_removed:
        warnings.append(
            "Duplicate detection used every cleaned column. Rows sharing only an ID or one field were retained."
        )
    if summary.invalid_values:
        warnings.append(
            "Some values could not be converted within the selected validity threshold. "
            "They were left empty in the cleaned column and listed with their original values in the cleaning report."
        )
    if summary.category_groups_detected and not options.normalize_category_case:
        warnings.append(
            "Possible inconsistent category spellings were detected but not changed. "
            "Review them in the report or enable category case normalization."
        )
    if options.day_first == "auto":
        warnings.append(
            "Ambiguous numeric dates were interpreted day-first (DD/MM/YYYY) by default. "
            "Choose the date order explicitly if the source uses MM/DD/YYYY."
        )
    if summary.empty_columns_removed:
        warnings.append("Empty columns were removed; their names are in the audit log.")
    if summary.empty_rows_removed:
        warnings.append("Fully empty rows were removed; the count is in the audit log.")

    return CleaningResult(
        table_name=table_name,
        raw=raw,
        cleaned=work,
        options=options,
        summary=summary,
        audit=audit,
        profile=profile,
        column_meta=column_meta,
        conversions=conversions,
        invalid_values=invalid_values,
        category_groups=category_groups,
        duplicates_removed_frame=duplicates_frame,
        column_renames=column_renames,
        warnings=_dedupe(warnings),
    )


def clean_tables(
    tables: dict[str, pd.DataFrame],
    options: CleaningOptions | None = None,
) -> dict[str, CleaningResult]:
    """Clean several tables with the same options (used for multi-sheet files)."""

    return {
        name: clean_dataframe(frame, options, table_name=name)
        for name, frame in tables.items()
    }


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
def standardize_column_name(name: Any) -> str:
    """Convert a header to a predictable snake_case name."""

    text = str(name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ").replace("%", " pct ").replace("#", " number ")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        return "column"
    if text[0].isdigit():
        return f"col_{text}"
    return text


def _unique_column_names(names: Iterable[str]) -> list[str]:
    used: set[str] = set()
    result: list[str] = []
    for name in names:
        base = name or "column"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------
def _detect_category_groups(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    categorical_columns: list[str] = []
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty or series.dtype != object:
            continue
        if looks_like_identifier_name(str(column)):
            continue
        unique_values = series.astype(str).unique()
        if not (2 <= len(unique_values) <= 50):
            continue
        avg_length = float(series.astype(str).str.len().mean())
        if avg_length > 60:
            continue
        categorical_columns.append(str(column))

        counts: dict[str, dict[str, int]] = {}
        for value in series.astype(str):
            if not value.strip():
                continue
            key = value.casefold()
            counts.setdefault(key, {})
            counts[key][value] = counts[key].get(value, 0) + 1
        inconsistent = []
        for key, variants in counts.items():
            if len(variants) > 1:
                total = sum(variants.values())
                canonical = sorted(variants.items(), key=lambda item: (-item[1], item[0]))[0][0]
                inconsistent.append(
                    {
                        "normalized": key,
                        "variants": [
                            {"value": value, "count": count}
                            for value, count in sorted(variants.items(), key=lambda item: (-item[1], item[0]))
                        ],
                        "total": total,
                        "canonical": canonical,
                        "canonical_count": variants[canonical],
                    }
                )
        if inconsistent:
            groups.append({"column": str(column), "groups": inconsistent})
    return groups, categorical_columns


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _count_missing(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    missing = int(frame.isna().sum().sum())
    for column in frame.columns:
        if frame[column].dtype == object:
            missing += int(
                frame[column]
                .map(lambda value: isinstance(value, str) and not value.strip())
                .sum()
            )
    return missing


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _series_kind_label(series: pd.Series) -> str:
    from pandas.api import types as ptypes

    if ptypes.is_datetime64_any_dtype(series):
        return "Date"
    if ptypes.is_bool_dtype(series):
        return "Boolean"
    if ptypes.is_numeric_dtype(series):
        return "Number"
    return "Text"


def _kind_label(kind: str) -> str:
    return {
        ColumnKind.NUMERIC: "Number",
        ColumnKind.CURRENCY: "Currency",
        ColumnKind.PERCENTAGE: "Percentage",
        ColumnKind.DATE: "Date",
        ColumnKind.DATETIME: "Date & time",
        ColumnKind.BOOLEAN: "Boolean",
        ColumnKind.IDENTIFIER: "Identifier (text)",
        ColumnKind.TEXT: "Text",
        ColumnKind.MIXED: "Mixed / text",
        ColumnKind.EMPTY: "Empty",
    }.get(kind, kind.title())


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
