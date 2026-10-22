"""Column type inference, data profiling, and analytics recommendations.

This module deliberately separates *detection* from *transformation*. The
cleaner asks for a :class:`ConversionOutcome` and decides whether to apply it.
Nothing here mutates the caller's DataFrame.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# ---------------------------------------------------------------------------
# Constants / small helpers
# ---------------------------------------------------------------------------
_CURRENCY_SYMBOLS = ("₹", "$", "€", "£", "¥", "₩", "₽", "¢", "₦", "₱")
_CURRENCY_CODES = (
    "INR", "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "SGD",
    "AED", "SAR", "ZAR", "NGN", "KES", "NZD", "HKD", "SEK", "NOK", "DKK",
)
_CURRENCY_CODE_RE = re.compile(
    r"(?i)(?<![A-Za-z])(?:" + "|".join(_CURRENCY_CODES) + r")(?![A-Za-z])"
)
_RS_RE = re.compile(r"(?i)(?<![A-Za-z])rs\.?(?![A-Za-z])")

_TRUE_VALUES = {"true", "yes", "y", "t", "1"}
_FALSE_VALUES = {"false", "no", "n", "f", "0"}
_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
_ID_TOKEN_LAST = {"id", "code", "no", "number", "num", "key", "sku", "isbn", "ean", "upc", "guid", "uuid"}
_ID_TOKENS = {
    "phone", "mobile", "telephone", "fax", "zip", "postal", "pincode", "pin",
    "account", "acct", "iban", "ifsc", "swift", "passport", "license", "licence",
    "roll", "aadhaar", "aadhar", "pan", "ssn", "vat", "gst", "ein", "registration",
}
_DATE_NAME_TOKENS = {"date", "datetime", "timestamp", "created", "updated", "dob", "due", "invoice_date", "order_date"}
_TEXT_FIELD_TOKENS = {
    "name", "email", "address", "description", "notes", "note", "comment",
    "remarks", "title", "summary", "message", "url", "link", "filename",
}
_DATEY_RE = re.compile(r"^\s*\d{1,4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,4}")
_YEAR_END_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_DIGIT_ONLY_RE = re.compile(r"^\d+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")

_MEASURE_EXCLUDE_NAMES = {"year", "month", "day", "quarter", "week", "hour", "minute", "second"}

_DATE_FORMAT_POOL = {
    "iso": [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
    ],
    "day_first": [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%d-%m-%y",
    ],
    "month_first": [
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%m/%d/%y",
        "%m-%d-%y",
    ],
    "textual": [
        "%d %b %Y",
        "%d-%b-%Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %Y",
        "%B %Y",
    ],
}


class ColumnKind:
    """String constants for the kinds CleanFlow recognises."""

    NUMERIC = "numeric"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    TEXT = "text"
    EMPTY = "empty"
    MIXED = "mixed"


@dataclass
class ConversionSettings:
    """Controls for conservative type conversion."""

    parse_dates: bool = True
    convert_currency: bool = True
    convert_percentage: bool = True
    numeric_threshold: float = 0.95
    date_threshold: float = 0.95
    day_first: str = "auto"  # auto | day_first | month_first
    decimal_style: str = "auto"  # auto | dot | comma


@dataclass
class ConversionOutcome:
    """Result of analysing one raw column."""

    kind: str
    converted: pd.Series | None
    valid_count: int = 0
    invalid_examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def converted_count(self) -> int:
        return self.valid_count


@dataclass
class ColumnProfile:
    name: str
    kind: str
    role: str
    dtype: str
    non_null: int
    missing: int
    missing_pct: float
    unique: int
    sample: list[str] = field(default_factory=list)
    top_values: list[dict[str, Any]] = field(default_factory=list)
    minimum: Any = None
    maximum: Any = None
    mean: Any = None
    median: Any = None
    std: Any = None
    total: Any = None
    issues: list[str] = field(default_factory=list)
    currency_symbols: list[str] = field(default_factory=list)
    outlier_count: int = 0
    outlier_examples: list[dict[str, Any]] = field(default_factory=list)
    source_kind: str | None = None
    date_format: str | None = None
    ambiguous_dates: int = 0

    @property
    def is_numeric_measure(self) -> bool:
        return self.role == "measure" and self.kind in {
            ColumnKind.NUMERIC, ColumnKind.CURRENCY, ColumnKind.PERCENTAGE
        }

    @property
    def kind_label(self) -> str:
        labels = {
            ColumnKind.NUMERIC: "Number",
            ColumnKind.CURRENCY: "Currency",
            ColumnKind.PERCENTAGE: "Percentage",
            ColumnKind.DATE: "Date",
            ColumnKind.DATETIME: "Date & time",
            ColumnKind.BOOLEAN: "Yes / No",
            ColumnKind.IDENTIFIER: "Identifier (text)",
            ColumnKind.TEXT: "Text",
            ColumnKind.EMPTY: "Empty",
            ColumnKind.MIXED: "Mixed",
        }
        return labels.get(self.kind, self.kind.title())

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.name,
            "type": self.kind_label,
            "role": self.role,
            "dtype": self.dtype,
            "non_null": self.non_null,
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 2),
            "unique": self.unique,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "total": self.total,
            "issues": " | ".join(self.issues),
            "sample": " | ".join(self.sample),
            "currency": " ".join(self.currency_symbols),
            "outliers": self.outlier_count,
            "date_format": self.date_format,
            "ambiguous_dates": self.ambiguous_dates,
        }


@dataclass
class DataProfile:
    rows: int
    columns: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    memory_bytes: int
    column_profiles: list[ColumnProfile]

    @property
    def by_name(self) -> dict[str, ColumnProfile]:
        return {profile.name: profile for profile in self.column_profiles}

    @property
    def numeric_profiles(self) -> list[ColumnProfile]:
        return [p for p in self.column_profiles if p.is_numeric_measure]

    @property
    def date_profiles(self) -> list[ColumnProfile]:
        return [p for p in self.column_profiles if p.kind in {ColumnKind.DATE, ColumnKind.DATETIME}]

    @property
    def categorical_profiles(self) -> list[ColumnProfile]:
        return [p for p in self.column_profiles if p.role == "dimension"]

    def to_frame(self) -> pd.DataFrame:
        rows = [profile.to_dict() for profile in self.column_profiles]
        if not rows:
            return pd.DataFrame(
                columns=["column", "type", "role", "missing", "unique", "issues"]
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Values / display
# ---------------------------------------------------------------------------
def stringify_value(value: Any) -> str:
    """Convert a cell to a stable string without losing integer formatting."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.isoformat()
        return value.strftime("%Y-%m-%d %H:%M:%S") if value.time() != datetime.min.time() else value.strftime("%Y-%m-%d")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
        return repr(number)
    text = str(value)
    return text.strip()


def display_value(value: Any, max_length: int = 80) -> str:
    text = stringify_value(value)
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if number.is_integer() and abs(number) < 1e15:
        return f"{int(number):,}"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:,.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}M"
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def format_metric(value: Any, kind: str, currency_symbol: str | None = None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if kind == ColumnKind.PERCENTAGE:
        return f"{format_number(value)}%"
    if kind == ColumnKind.CURRENCY:
        prefix = f"{currency_symbol} " if currency_symbol else ""
        return f"{prefix}{format_number(value)}"
    return format_number(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _non_null_values(series: pd.Series) -> list[Any]:
    return [value for value in series.tolist() if not _is_missing(value)]


# ---------------------------------------------------------------------------
# Identifier detection
# ---------------------------------------------------------------------------
def _name_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(name).lower()) if token]


def _is_identifier_name(name: str) -> bool:
    tokens = _name_tokens(name)
    if not tokens:
        return False
    if tokens[-1] in _ID_TOKEN_LAST:
        return True
    if any(token in _ID_TOKENS for token in tokens):
        # "phone_type" is a category, not a phone number.
        if tokens[-1] in {"type", "category", "status", "label", "format"}:
            return False
        return True
    return False


def looks_like_identifier_name(name: str) -> bool:
    """Public wrapper used by the cleaning engine."""

    return _is_identifier_name(name)


def _looks_like_identifier_values(values: list[Any]) -> tuple[bool, str]:
    strings = [stringify_value(value) for value in values]
    non_empty = [value for value in strings if value]
    if not non_empty:
        return False, ""
    if any(re.match(r"^0\d+$", value) for value in non_empty):
        return True, "leading zeros detected"
    digit_phone = [
        value for value in non_empty
        if not _DATEY_RE.match(value)
        and _PHONE_RE.match(value)
        and len(re.sub(r"\D", "", value)) >= 7
    ]
    if digit_phone and (
        any(value.startswith("+") for value in digit_phone)
        or any(re.search(r"[\s()]", value) for value in digit_phone)
        or any(len(re.sub(r"\D", "", value)) >= 10 and re.search(r"[\s().-]", value) for value in digit_phone)
    ):
        return True, "phone-like formatting detected"
    digit_lengths = [
        len(re.sub(r"\D", "", value))
        for value in non_empty
        if _DIGIT_ONLY_RE.match(value)
    ]
    if digit_lengths and len(digit_lengths) == len(non_empty) and min(digit_lengths) >= 9:
        return True, "long digit strings detected"
    return False, ""


def _stringify_identifier(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return stringify_value(value)


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------
def parse_number(value: Any, decimal_style: str = "auto") -> dict[str, Any] | None:
    """Parse one numeric-ish cell.

    Returns ``{"number", "currency", "percent", "ambiguous_separator"}`` or
    ``None`` when the value is not a number.

    Percentages are returned as percentage points (``12.5%`` -> ``12.5``) so
    that dashboard readers do not see an unexpected 0.125. This policy is shown
    in the cleaning report and in the README.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return {"number": float(value), "currency": "", "percent": False, "ambiguous_separator": False}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return {"number": number, "currency": "", "percent": False, "ambiguous_separator": False}
        return None

    raw = str(value).strip()
    if not raw:
        return None

    percent = "%" in raw or "percent" in raw.lower()
    text = raw.replace("%", "")
    text = text.replace("percent", "").replace("Percent", "")
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("'", "")
    text = text.strip()

    currency_found: list[str] = []
    for symbol in _CURRENCY_SYMBOLS:
        if symbol in text:
            currency_found.append(symbol)
            text = text.replace(symbol, "")
    if _RS_RE.search(text):
        currency_found.append("Rs")
        text = _RS_RE.sub("", text)
    code_matches = _CURRENCY_CODE_RE.findall(text)
    if code_matches:
        currency_found.extend(code.upper() for code in code_matches)
        text = _CURRENCY_CODE_RE.sub("", text)

    text = text.strip().replace(" ", "")
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.startswith("+"):
        text = text[1:]

    if not re.search(r"\d", text):
        return None

    comma = text.rfind(",")
    dot = text.rfind(".")
    ambiguous = False

    if decimal_style in {"dot", "us"}:
        text = text.replace(",", "")
    elif decimal_style in {"comma", "eu"}:
        text = text.replace(".", "").replace(",", ".")
    elif comma != -1 and dot != -1:
        if comma > dot:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif comma != -1:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            text = text.replace(",", ".")
        elif len(parts) == 2:
            # "1,234" is ambiguous. Default to thousands, the most common
            # interpretation in English-language exports; the UI explains this.
            ambiguous = True
            text = text.replace(",", "")
        else:
            text = text.replace(",", "")
    elif dot != -1:
        parts = text.split(".")
        if len(parts) > 2 and all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)
            ambiguous = True

    if not _NUMERIC_RE.match(text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if negative:
        number = -abs(number)

    return {
        "number": number,
        "currency": " ".join(sorted(set(currency_found))),
        "percent": percent,
        "ambiguous_separator": ambiguous,
    }


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
def _might_be_date(values: list[Any], name: str) -> bool:
    if _is_identifier_name(name):
        return False
    name_hint = any(token in _DATE_NAME_TOKENS for token in _name_tokens(name))
    for value in values[:50]:
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return True
        text = stringify_value(value)
        if not text:
            continue
        if _DATEY_RE.match(text):
            return True
        lowered = text.lower()
        if any(month in lowered for month in ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")):
            return True
        if name_hint and re.match(r"^\d{4}-\d{2}-\d{2}", text):
            return True
    return False


def _date_evidence(values: list[Any]) -> tuple[int, int, int]:
    """Return (day_first_evidence, month_first_evidence, ambiguous_count)."""

    day_first = 0
    month_first = 0
    ambiguous = 0
    for value in values:
        text = stringify_value(value)
        match = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", text)
        if not match:
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        if first > 12 and second <= 12:
            day_first += 1
        elif second > 12 and first <= 12:
            month_first += 1
        elif first <= 12 and second <= 12:
            ambiguous += 1
    return day_first, month_first, ambiguous


def _format_order(day_first: str) -> list[str]:
    return list(_DATE_FORMAT_POOL["iso"]) + list(
        _DATE_FORMAT_POOL["day_first" if day_first == "day_first" else "month_first"]
    ) + list(_DATE_FORMAT_POOL["textual"])


def _parse_dates(series: pd.Series, name: str, settings: ConversionSettings) -> ConversionOutcome | None:
    if not settings.parse_dates:
        return None
    non_null_positions = [position for position, value in enumerate(series.tolist()) if not _is_missing(value)]
    if not non_null_positions:
        return None
    values = [series.iloc[position] for position in non_null_positions]
    if not _might_be_date(values, name):
        return None

    # Already a datetime dtype: accept it directly.
    if ptypes.is_datetime64_any_dtype(series):
        converted = pd.to_datetime(series, errors="coerce")
        if getattr(converted.dt, "tz", None) is not None:
            converted = converted.dt.tz_localize(None)
        return ConversionOutcome(
            kind=ColumnKind.DATE if _is_date_only(converted) else ColumnKind.DATETIME,
            converted=converted,
            valid_count=int(converted.notna().sum()),
            metadata={"format": "Excel date value"},
        )

    day_first_evidence, month_first_evidence, ambiguous_count = _date_evidence(values)
    configured = settings.day_first if settings.day_first in {"day_first", "month_first"} else None
    if configured:
        order = configured
    elif day_first_evidence and not month_first_evidence:
        order = "day_first"
    elif month_first_evidence and not day_first_evidence:
        order = "month_first"
    else:
        # Documented default: day-first, because the product is launched for
        # Indian/UK-style business data. The user can switch to month-first.
        order = "day_first"

    formats = _format_order(order)
    strings = series.map(stringify_value)

    sample_positions = non_null_positions[:2000]
    sample = [series.iloc[position] for position in sample_positions]
    sample_strings = pd.Series([stringify_value(value) for value in sample])
    ranked: list[tuple[int, int, str]] = []
    for index, fmt in enumerate(formats):
        try:
            parsed = pd.to_datetime(sample_strings, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        count = int(parsed.notna().sum())
        if count:
            ranked.append((count, index, fmt))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    ranked_formats = [fmt for _, _, fmt in ranked[:8]]

    converted = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    used_formats: list[str] = []
    pending = strings.copy()
    for fmt in ranked_formats:
        if converted.notna().all():
            break
        mask = converted.isna() & pending.ne("")
        if not mask.any():
            break
        try:
            parsed = pd.to_datetime(pending[mask], format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        if parsed.notna().any():
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
            converted.loc[mask] = parsed
            used_formats.append(fmt)

    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_localize(None)

    total = len(non_null_positions)
    valid = int(converted.notna().sum())
    coverage = valid / total if total else 0.0
    if coverage < settings.date_threshold:
        return None

    invalid_mask = converted.isna() & series.map(lambda value: not _is_missing(value))
    invalid_examples = _invalid_examples(series, invalid_mask, kind="date")
    issues: list[str] = []
    invalid_total = total - valid
    if invalid_total:
        issues.append(
            f"{invalid_total} value(s) could not be parsed as dates; "
            "they are listed in the cleaning report and left empty."
        )
    if ambiguous_count:
        issues.append(
            f"{ambiguous_count} ambiguous day/month value(s) were interpreted as "
            f"{'day-first' if order == 'day_first' else 'month-first'} (DD/MM/YYYY). "
            "Change the date order option before cleaning if this is wrong."
        )
    if day_first_evidence and month_first_evidence:
        issues.append("The column mixes day-first and month-first dates; review the parsed values carefully.")

    kind = ColumnKind.DATE if _is_date_only(converted) else ColumnKind.DATETIME
    return ConversionOutcome(
        kind=kind,
        converted=converted,
        valid_count=valid,
        invalid_examples=invalid_examples,
        metadata={
            "date_order": order,
            "formats": used_formats[:4],
            "ambiguous_dates": ambiguous_count,
            "format": used_formats[0] if used_formats else None,
        },
        issues=issues,
    )


def _is_date_only(series: pd.Series) -> bool:
    try:
        times = series.dropna().dt.time
        return all(time == datetime.min.time() for time in times)
    except (AttributeError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Conversion planning
# ---------------------------------------------------------------------------
def analyse_and_convert(
    series: pd.Series,
    name: str,
    settings: ConversionSettings | None = None,
) -> ConversionOutcome:
    """Analyse one column and propose a converted Series.

    Conversion is all-or-nothing per column with a configurable validity
    threshold. If too many values fail, the column stays as text so meaningful
    data is never silently discarded.
    """

    settings = settings or ConversionSettings()
    non_null = _non_null_values(series)
    total = len(non_null)
    if total == 0:
        return ConversionOutcome(kind=ColumnKind.EMPTY, converted=None)

    # 1. Identifiers. This runs before numeric conversion to preserve leading
    # zeros, phone formatting, and long account numbers.
    name_identifier = _is_identifier_name(name)
    value_identifier, reason = _looks_like_identifier_values(non_null)
    if name_identifier or value_identifier:
        converted = series.map(_stringify_identifier)
        metadata = {"identifier_reason": reason or "column name suggests an identifier"}
        issues: list[str] = []
        if any(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) for value in non_null):
            issues.append(
                "Kept as text to preserve identifiers. If Excel stored this as a number, "
                "leading zeros were already lost before upload."
            )
        return ConversionOutcome(
            kind=ColumnKind.IDENTIFIER,
            converted=converted,
            valid_count=int(converted.notna().sum()),
            metadata=metadata,
            issues=issues,
        )

    # 2. Boolean words (not 0/1, which are better treated as numbers).
    bool_strings = [stringify_value(value).lower() for value in non_null]
    bool_ratio = sum(1 for value in bool_strings if value in _BOOL_WORDS) / total
    has_words = any(value in {"true", "false", "yes", "no", "y", "n", "t", "f"} for value in bool_strings)
    if bool_ratio >= settings.numeric_threshold and has_words:
        converted = series.map(
            lambda value: _to_bool(value)
        )
        return ConversionOutcome(
            kind=ColumnKind.BOOLEAN,
            converted=converted,
            valid_count=int(converted.notna().sum()),
            metadata={"mapping": "true/yes/y/t -> TRUE; false/no/n/f -> FALSE"},
        )

    # 3. Numbers, currency, and percentages.
    parsed: list[dict[str, Any] | None] = []
    for value in series.tolist():
        if _is_missing(value):
            parsed.append(None)
        else:
            parsed.append(parse_number(value, decimal_style=settings.decimal_style))
    success = [item for item in parsed if item is not None]
    numeric_ratio = len(success) / total

    if success and numeric_ratio >= settings.numeric_threshold:
        currencies = sorted({item["currency"] for item in success if item["currency"]})
        percent_count = sum(1 for item in success if item["percent"])
        percentage_name_hint = any(
            token in {"pct", "percent", "percentage"} for token in _name_tokens(name)
        )
        is_percentage = (
            (percent_count / len(success) >= 0.5 or percentage_name_hint)
            and settings.convert_percentage
            and not currencies
        )
        is_currency = bool(currencies) and settings.convert_currency
        kind = (
            ColumnKind.PERCENTAGE if is_percentage
            else ColumnKind.CURRENCY if is_currency
            else ColumnKind.NUMERIC
        )

        numbers = [
            float(item["number"]) if item is not None else np.nan
            for item in parsed
        ]
        numeric_series = pd.Series(numbers, index=series.index, dtype="float64")
        converted = _maybe_integer_series(numeric_series)
        invalid_mask = numeric_series.isna() & series.map(lambda value: not _is_missing(value))
        invalid_examples = _invalid_examples(series, invalid_mask, kind="number")

        issues = []
        if invalid_examples:
            issues.append(
                f"{int(invalid_mask.sum())} non-numeric value(s) were left empty and are listed in the cleaning report."
            )
        if len(currencies) > 1:
            issues.append(
                "Multiple currency indicators were detected ("
                + ", ".join(currencies)
                + "). Totals across mixed currencies are not meaningful; CleanFlow will not sum this column in KPIs."
            )
        if any(item["ambiguous_separator"] for item in success):
            issues.append(
                "Some values used ambiguous thousands separators (for example 1,234); "
                "they were read with the selected decimal style. Review the converted values."
            )
        if is_percentage:
            issues.append("Percentages are stored as percentage points: 12.5% becomes 12.5.")

        metadata = {
            "currencies": currencies,
            "percent_values": percent_count,
            "source_kind": "percentage" if is_percentage else "currency" if is_currency else "numeric",
        }
        return ConversionOutcome(
            kind=kind,
            converted=converted,
            valid_count=int(converted.notna().sum()),
            invalid_examples=invalid_examples,
            metadata=metadata,
            issues=issues,
        )

    # 4. Dates. Run after numbers so that year-like columns stay numeric.
    date_outcome = _parse_dates(series, name, settings)
    if date_outcome is not None:
        return date_outcome

    # 5. Mixed / text. Report why numeric-looking data was not converted.
    issues = []
    if success and numeric_ratio < settings.numeric_threshold:
        issues.append(
            f"Only {numeric_ratio:.0%} of values parsed as numbers. "
            "The column was kept as text to avoid deleting the remaining values."
        )
    if settings.parse_dates and _might_be_date(non_null, name):
        issues.append(
            "Some values look like dates, but too few could be parsed consistently. "
            "The column was kept as text. Check the date order option or fix the inconsistent values."
        )
    if numeric_ratio >= 0.5 and numeric_ratio < settings.numeric_threshold:
        kind = ColumnKind.MIXED
    else:
        kind = ColumnKind.TEXT
    return ConversionOutcome(kind=kind, converted=None, valid_count=0, issues=issues)


def _to_bool(value: Any) -> Any:
    if _is_missing(value):
        return None
    lowered = stringify_value(value).lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return None


def _maybe_integer_series(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series
    if all(float(value).is_integer() for value in non_null):
        if non_null.abs().max() < 9_007_199_254_740_992:
            return series.astype("Int64")
    return series


def _invalid_examples(series: pd.Series, mask: pd.Series, *, kind: str, limit: int = 10) -> list[dict[str, Any]]:
    if not mask.any():
        return []
    positions = [index for index, value in zip(series.index, mask.tolist()) if value]
    examples: list[dict[str, Any]] = []
    for position in positions[:limit]:
        raw = series.loc[position]
        examples.append(
            {
                "row": _row_number(position),
                "value": display_value(raw, max_length=60),
                "problem": f"not a valid {kind}",
            }
        )
    return examples


def _row_number(index: Any) -> Any:
    if isinstance(index, (int, np.integer)) and not isinstance(index, bool):
        return int(index) + 2  # +1 for zero-based index, +1 for the header row
    return stringify_value(index)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def profile_dataframe(
    df: pd.DataFrame,
    kind_hints: dict[str, dict[str, Any]] | None = None,
) -> DataProfile:
    """Profile a DataFrame without modifying it.

    ``kind_hints`` can carry the cleaner's conversion metadata so that currency
    and percentage columns retain their labels after conversion to numbers.
    """

    row_count = int(df.shape[0])
    column_count = int(df.shape[1])
    missing_cells = int(df.isna().sum().sum()) if column_count else 0
    cells = row_count * column_count
    missing_pct = (missing_cells / cells * 100.0) if cells else 0.0
    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        duplicate_rows = 0
    try:
        memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        memory_bytes = 0

    hints = kind_hints or {}
    profiles = [
        _profile_column(df[column], str(column), hints.get(str(column), {}), row_count)
        for column in df.columns
    ]
    return DataProfile(
        rows=row_count,
        columns=column_count,
        missing_cells=missing_cells,
        missing_pct=round(missing_pct, 2),
        duplicate_rows=duplicate_rows,
        memory_bytes=memory_bytes,
        column_profiles=profiles,
    )


def _profile_column(series: pd.Series, name: str, hint: dict[str, Any], row_count: int) -> ColumnProfile:
    non_null_series = series.dropna()
    missing = int(series.isna().sum())
    non_null = int(len(non_null_series))
    unique = int(non_null_series.nunique(dropna=True))
    missing_pct = (missing / row_count * 100.0) if row_count else 0.0

    source_kind = hint.get("kind")
    kind, role = _infer_profile_kind(series, name, hint)
    issues = list(hint.get("issues", []))

    sample: list[str] = []
    for value in non_null_series.head(20).tolist():
        text = display_value(value)
        if text and text not in sample:
            sample.append(text)
        if len(sample) >= 5:
            break

    top_values: list[dict[str, Any]] = []
    if 0 < unique <= 50 and role in {"dimension", "identifier"}:
        try:
            counts = non_null_series.astype(str).value_counts().head(8)
            top_values = [{"value": str(index), "count": int(count)} for index, count in counts.items()]
        except Exception:
            top_values = []

    minimum = maximum = mean = median = std = total = None
    outlier_count = 0
    outlier_examples: list[dict[str, Any]] = []
    if kind in {ColumnKind.NUMERIC, ColumnKind.CURRENCY, ColumnKind.PERCENTAGE}:
        numeric = pd.to_numeric(non_null_series, errors="coerce").dropna()
        if not numeric.empty:
            minimum = _json_number(numeric.min())
            maximum = _json_number(numeric.max())
            mean = _json_number(numeric.mean())
            median = _json_number(numeric.median())
            std = _json_number(numeric.std()) if len(numeric) > 1 else 0.0
            total = _json_number(numeric.sum())
            outlier_count, outlier_examples = _detect_outliers(series, numeric)

    if unique == 1 and non_null:
        issues.append("Constant column: every non-empty row has the same value.")
    if kind in {ColumnKind.TEXT, ColumnKind.MIXED} and unique > max(50, row_count * 0.6):
        issues.append("High-cardinality text; likely free text rather than a category.")
    if outlier_count:
        issues.append(
            f"{outlier_count} possible outlier(s) detected with the IQR rule. "
            "They were not removed; review them before excluding."
        )
    if hint.get("currencies") and len(hint["currencies"]) > 1:
        issues.append("Mixed currency indicators; do not sum this column across currencies.")
    if hint.get("format") and kind in {ColumnKind.DATE, ColumnKind.DATETIME}:
        issues.append(f"Dates parsed using format {hint['format']}.")

    return ColumnProfile(
        name=name,
        kind=kind,
        role=role,
        dtype=str(series.dtype),
        non_null=non_null,
        missing=missing,
        missing_pct=round(missing_pct, 2),
        unique=unique,
        sample=sample,
        top_values=top_values,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        median=median,
        std=std,
        total=total,
        issues=_dedupe(issues),
        currency_symbols=list(hint.get("currencies", [])),
        outlier_count=outlier_count,
        outlier_examples=outlier_examples,
        source_kind=source_kind,
        date_format=hint.get("format"),
        ambiguous_dates=int(hint.get("ambiguous_dates", 0) or 0),
    )


def _infer_profile_kind(series: pd.Series, name: str, hint: dict[str, Any]) -> tuple[str, str]:
    hinted = hint.get("kind")
    if hinted == ColumnKind.IDENTIFIER:
        return ColumnKind.IDENTIFIER, "identifier"
    if hinted == ColumnKind.BOOLEAN:
        return ColumnKind.BOOLEAN, "dimension"
    if hinted in {ColumnKind.DATE, ColumnKind.DATETIME}:
        return hinted, "date"
    if hinted == ColumnKind.PERCENTAGE:
        return ColumnKind.PERCENTAGE, "measure"
    if hinted == ColumnKind.CURRENCY:
        return ColumnKind.CURRENCY, "measure"
    if hinted == ColumnKind.NUMERIC:
        role = "measure"
        tokens = _name_tokens(name)
        if tokens and (tokens[-1] in _MEASURE_EXCLUDE_NAMES or name.lower() in _MEASURE_EXCLUDE_NAMES):
            role = "dimension"
        return ColumnKind.NUMERIC, role

    if ptypes.is_datetime64_any_dtype(series):
        return (ColumnKind.DATE if _is_date_only(series) else ColumnKind.DATETIME), "date"
    if ptypes.is_bool_dtype(series):
        return ColumnKind.BOOLEAN, "dimension"
    if ptypes.is_numeric_dtype(series):
        role = "dimension" if name.lower() in _MEASURE_EXCLUDE_NAMES else "measure"
        return ColumnKind.NUMERIC, role
    if hinted == ColumnKind.TEXT:
        return ColumnKind.TEXT, _text_role(series, name)
    if hinted == ColumnKind.MIXED:
        return ColumnKind.MIXED, "text"
    if series.dropna().empty:
        return ColumnKind.EMPTY, "empty"

    values = series.dropna().tolist()
    if _is_identifier_name(name):
        return ColumnKind.IDENTIFIER, "identifier"

    # Lightweight raw-profile inference. The cleaning engine does the
    # conservative, explicit conversion; this is only for preview labels.
    sample = values[:100]
    numeric_success = sum(1 for value in sample if parse_number(value) is not None)
    if numeric_success / max(len(sample), 1) >= 0.9:
        if any("%" in stringify_value(value) for value in sample):
            return ColumnKind.PERCENTAGE, "measure"
        currencies = [parse_number(value) or {} for value in sample]
        if any(item.get("currency") for item in currencies):
            return ColumnKind.CURRENCY, "measure"
        return ColumnKind.NUMERIC, "measure"
    if _might_be_date(sample, name) and all(_DATEY_RE.match(stringify_value(value)) or not stringify_value(value) for value in sample[:20]):
        return ColumnKind.DATE, "date"
    distinct_sample = len(set(map(str, sample)))
    return ColumnKind.TEXT, _text_role(series, name)


def _text_role(series: pd.Series, name: str) -> str:
    """Decide whether a text column is a useful grouping dimension."""

    tokens = _name_tokens(name)
    if any(token in _TEXT_FIELD_TOKENS for token in tokens):
        return "text"
    try:
        values = series.dropna().astype(str)
    except Exception:
        return "text"
    if values.empty:
        return "text"
    unique = int(values.nunique())
    if not (2 <= unique <= 50):
        return "text"
    if float(values.str.len().mean()) > 40:
        return "text"
    return "dimension"


def _detect_outliers(series: pd.Series, numeric: pd.Series) -> tuple[int, list[dict[str, Any]]]:
    if len(numeric) < 8:
        return 0, []
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return 0, []
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (numeric < lower) | (numeric > upper)
    count = int(mask.sum())
    if count == 0:
        return 0, []
    examples = []
    for position in numeric[mask].head(5).index.tolist():
        examples.append(
            {
                "row": _row_number(position),
                "value": display_value(series.loc[position]),
            }
        )
    return count, examples


def _json_number(value: Any) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 9_007_199_254_740_992:
        return int(number)
    return round(number, 6)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# KPI and chart recommendations
# ---------------------------------------------------------------------------
@dataclass
class Kpi:
    label: str
    value: Any
    display: str
    hint: str = ""
    kind: str = "number"
    currency: str | None = None


def recommend_kpis(
    df: pd.DataFrame,
    profile: DataProfile,
    *,
    duplicates_removed: int | None = None,
    max_measures: int = 1,
) -> list[Kpi]:
    """Build KPI cards that are meaningful for this dataset."""

    kpis = [
        Kpi("Total records", int(df.shape[0]), format_number(df.shape[0]), "Rows in the cleaned dataset"),
        Kpi("Columns", int(df.shape[1]), format_number(df.shape[1]), "Fields available for analysis"),
        Kpi(
            "Numeric columns",
            len(profile.numeric_profiles),
            format_number(len(profile.numeric_profiles)),
            "Numeric measures detected in the current view",
        ),
        Kpi(
            "Missing cells",
            int(profile.missing_cells),
            format_number(profile.missing_cells),
            f"{profile.missing_pct:.1f}% of all cells",
            kind="number",
        ),
    ]

    if duplicates_removed is not None:
        kpis.append(
            Kpi(
                "Duplicates removed",
                int(duplicates_removed),
                format_number(duplicates_removed),
                "Rows removed before this dashboard was created",
            )
        )

    measures = [p for p in profile.column_profiles if p.is_numeric_measure and p.unique > 1]
    # Prefer high-value measures: currency first, then numeric, then percentage.
    def rank(profile: ColumnProfile) -> tuple[int, int]:
        priority = {ColumnKind.CURRENCY: 0, ColumnKind.NUMERIC: 1, ColumnKind.PERCENTAGE: 2}.get(profile.kind, 3)
        return priority, -abs(float(profile.total or 0))

    dimensions = [
        item for item in profile.categorical_profiles
        if item.kind != ColumnKind.IDENTIFIER and 2 <= item.unique <= 100
    ]
    if dimensions:
        category = min(dimensions, key=lambda item: item.unique)
        kpis.append(
            Kpi(
                f"Distinct {category.name}",
                category.unique,
                format_number(category.unique),
                f"Non-empty categories in {category.name}",
            )
        )

    for measure_index, column_profile in enumerate(sorted(measures, key=rank)[:max_measures]):
        kind = column_profile.kind
        mixed_currency = len(column_profile.currency_symbols) > 1
        symbol = column_profile.currency_symbols[0] if len(column_profile.currency_symbols) == 1 else None
        if kind == ColumnKind.PERCENTAGE:
            kpis.append(
                Kpi(
                    f"Average {column_profile.name}",
                    column_profile.mean,
                    format_metric(column_profile.mean, kind),
                    f"Mean of {column_profile.name} (values are percentage points)",
                    kind=kind,
                )
            )
        elif kind == ColumnKind.CURRENCY and mixed_currency:
            kpis.append(
                Kpi(
                    f"Average {column_profile.name}",
                    column_profile.mean,
                    format_metric(column_profile.mean, ColumnKind.NUMERIC),
                    "Mixed currencies detected; total is intentionally not shown",
                    kind=ColumnKind.NUMERIC,
                )
            )
        else:
            kpis.append(
                Kpi(
                    f"Total {column_profile.name}",
                    column_profile.total,
                    format_metric(column_profile.total, kind, symbol),
                    f"Sum of non-missing values in {column_profile.name}",
                    kind=kind,
                    currency=symbol,
                )
            )
        if measure_index == 0:
            kpis.append(
                Kpi(
                    f"Average {column_profile.name}",
                    column_profile.mean,
                    format_metric(column_profile.mean, kind, symbol),
                    f"Mean of non-missing values in {column_profile.name}",
                    kind=kind,
                    currency=symbol,
                )
            )
            kpis.append(
                Kpi(
                    f"Range {column_profile.name}",
                    None,
                    f"{format_metric(column_profile.minimum, kind, symbol)} – {format_metric(column_profile.maximum, kind, symbol)}",
                    f"Minimum and maximum non-missing values in {column_profile.name}",
                    kind="text",
                )
            )

    date_profile = profile.date_profiles[0] if profile.date_profiles else None
    if date_profile and date_profile.minimum and date_profile.maximum:
        kpis.append(
            Kpi(
                "Date range",
                None,
                f"{date_profile.minimum} → {date_profile.maximum}",
                f"Range of {date_profile.name}",
                kind="text",
            )
        )

    return kpis


@dataclass
class ChartSpec:
    id: str
    kind: str
    title: str
    reason: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str | None = None
    period: str | None = None
    limit: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


def recommend_charts(df: pd.DataFrame, profile: DataProfile, *, max_charts: int = 6) -> list[ChartSpec]:
    """Suggest charts only when the data supports them."""

    if df.empty:
        return []
    specs: list[ChartSpec] = []
    measures = [
        p for p in profile.numeric_profiles if p.unique > 1
    ]
    measures.sort(
        key=lambda item: (
            {ColumnKind.CURRENCY: 0, ColumnKind.NUMERIC: 1, ColumnKind.PERCENTAGE: 2}.get(item.kind, 3),
            -abs(float(item.total or 0)),
        )
    )
    dates = profile.date_profiles
    dimensions = [
        p for p in profile.categorical_profiles
        if 2 <= p.unique <= 12
        and p.missing_pct < 50
        and p.kind not in {ColumnKind.IDENTIFIER}
        and (not p.sample or max(len(value) for value in p.sample) <= 40)
    ]

    primary_measure = measures[0] if measures else None
    primary_date = dates[0] if dates else None

    if primary_date and len(df) >= 3:
        y = primary_measure.name if primary_measure else None
        agg = None
        if primary_measure:
            agg = "mean" if primary_measure.kind == ColumnKind.PERCENTAGE else "sum"
        specs.append(
            ChartSpec(
                id="trend",
                kind="line",
                title=f"{primary_date.name}: trend over time",
                reason="A valid date column enables a time-based view.",
                x=primary_date.name,
                y=y,
                agg=agg,
                period="auto",
            )
        )

    for dimension in dimensions[:3]:
        if primary_measure:
            specs.append(
                ChartSpec(
                    id=f"bar_{dimension.name}",
                    kind="bar",
                    title=f"{primary_measure.name} by {dimension.name}",
                    reason="A categorical column with a numeric measure supports comparison.",
                    x=dimension.name,
                    y=primary_measure.name,
                    agg="mean" if primary_measure.kind == ColumnKind.PERCENTAGE else "sum",
                    limit=10,
                )
            )
        else:
            specs.append(
                ChartSpec(
                    id=f"donut_{dimension.name}",
                    kind="donut",
                    title=f"Record share by {dimension.name}",
                    reason="A category with a manageable number of values supports a distribution chart.",
                    x=dimension.name,
                    agg="count",
                    limit=8,
                )
            )
        if len(specs) >= max_charts:
            return specs[:max_charts]

    if primary_measure and len(df) >= 5:
        specs.append(
            ChartSpec(
                id="histogram",
                kind="histogram",
                title=f"Distribution of {primary_measure.name}",
                reason="A numeric measure can be checked for skew and unusual values.",
                x=primary_measure.name,
                limit=40,
            )
        )

    numeric_for_corr = _correlation_candidates(profile)
    if len(numeric_for_corr) >= 3 and len(df) >= 10 and len(specs) < max_charts:
        specs.append(
            ChartSpec(
                id="correlation",
                kind="heatmap",
                title="Correlation between numeric fields",
                reason="Three or more numeric fields allow a correlation heatmap.",
                extra={"columns": [p.name for p in numeric_for_corr[:8]]},
            )
        )

    if len(numeric_for_corr) >= 2 and len(df) >= 20 and len(specs) < max_charts:
        pair = _best_numeric_pair(df, numeric_for_corr)
        if pair:
            specs.append(
                ChartSpec(
                    id="scatter",
                    kind="scatter",
                    title=f"{pair[1]} vs {pair[0]}",
                    reason="Two numeric measures support a relationship view.",
                    x=pair[0],
                    y=pair[1],
                    limit=3000,
                )
            )

    return specs[:max_charts]


def _correlation_candidates(profile: DataProfile) -> list[ColumnProfile]:
    candidates = []
    for item in profile.numeric_profiles:
        if item.unique > 2 and len(item.currency_symbols) <= 1:
            candidates.append(item)
    return candidates


def _best_numeric_pair(df: pd.DataFrame, profiles: list[ColumnProfile]) -> tuple[str, str] | None:
    columns = [profile.name for profile in profiles[:6]]
    if len(columns) < 2:
        return None
    try:
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        correlation = numeric.corr().abs()
    except Exception:
        return None
    best: tuple[float, str, str] | None = None
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            try:
                value = float(correlation.loc[first, second])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and (best is None or value > best[0]):
                best = (value, first, second)
    if not best or best[0] < 0.15:
        return None
    return best[1], best[2]
