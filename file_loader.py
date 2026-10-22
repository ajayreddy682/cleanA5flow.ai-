"""Load Excel, CSV, JSON, and PDF uploads into tabular data.

Everything happens in memory from the uploaded bytes. Original bytes are kept
on the returned ``LoadedFile`` so the application can offer a separate
"download original" action without ever writing the upload to a shared folder.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from utils.security import safe_error_message, scan_pdf_bytes
from utils.validators import (
    MAX_EXCEL_SHEETS,
    MAX_PDF_PAGES,
    ValidationError,
    sanitize_filename,
    validate_dataframe_limits,
    validate_upload,
)

LOGGER = logging.getLogger("cleanflow.loader")

_NUMBERISH_RE = re.compile(r"^[\s+-]?[\d.,%$€£¥₹]+$")
_SUPPORTED_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class FileLoadError(ValueError):
    """A user-facing error raised while parsing a validated upload."""


@dataclass
class TableData:
    """One selectable table extracted from an upload."""

    name: str
    dataframe: pd.DataFrame
    source_type: str = "table"  # excel_sheet | csv | json | pdf_table | pdf_text
    page: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(self.dataframe.shape[0])

    @property
    def columns(self) -> int:
        return int(self.dataframe.shape[1])

    @property
    def label(self) -> str:
        return f"{self.name}  ·  {self.rows:,} rows × {self.columns:,} columns"


@dataclass
class LoadedFile:
    """The validated upload and all tables that could be extracted from it."""

    filename: str
    file_type: str
    size_bytes: int
    original_bytes: bytes
    tables: list[TableData]
    warnings: list[str] = field(default_factory=list)
    encoding: str | None = None

    @property
    def primary_table(self) -> TableData:
        return self.tables[0]

    def table_by_name(self, name: str) -> TableData | None:
        return next((table for table in self.tables if table.name == name), None)

    @property
    def total_rows(self) -> int:
        return sum(table.rows for table in self.tables)


def load_uploaded_file(filename: str | None, data: bytes, *, attempt_ocr: bool = False) -> LoadedFile:
    """Validate and parse an upload.

    Args:
        filename: Original upload name (used for type detection and display only).
        data: Raw bytes.
        attempt_ocr: If True, a scanned PDF with no text layer is sent through
            the optional local OCR pathway when Tesseract/Poppler are available.

    Raises:
        ValidationError: for unsupported, unsafe, empty, or oversized files.
        FileLoadError: for structurally readable files that contain no usable table.
    """

    validation = validate_upload(filename, data)
    warnings = list(validation.warnings)
    encoding: str | None = None
    tables: list[TableData] = []

    try:
        if validation.file_type == "csv":
            table, encoding, csv_warnings = _load_csv(data)
            table.name = sanitize_filename(Path(validation.filename).stem, "CSV data")
            tables = [table]
            warnings.extend(csv_warnings)
        elif validation.file_type in {"xlsx", "xls"}:
            tables, excel_warnings = _load_excel(data, validation.file_type)
            warnings.extend(excel_warnings)
        elif validation.file_type == "json":
            table_list, encoding, json_warnings = _load_json(data)
            tables = table_list
            warnings.extend(json_warnings)
        elif validation.file_type == "pdf":
            tables, pdf_warnings, used_ocr = _load_pdf(data, attempt_ocr=attempt_ocr)
            warnings.extend(pdf_warnings)
            if used_ocr:
                warnings.append(
                    "OCR was used because this PDF had no usable text layer. "
                    "OCR text can contain recognition errors; review the extracted table before relying on it."
                )
        else:  # pragma: no cover - validation prevents this
            raise FileLoadError("This file type is not supported.")
    except ValidationError:
        raise
    except FileLoadError:
        raise
    except ImportError as exc:
        LOGGER.exception("Optional reader dependency missing")
        raise FileLoadError(
            "A reader required for this file type is not installed. "
            "Install the packages from requirements.txt and try again."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary
        LOGGER.exception("Failed to parse upload")
        raise FileLoadError(safe_error_message(exc)) from exc

    if not tables:
        raise FileLoadError(
            "No usable data table was found in this file. "
            "If it is a scanned PDF, enable OCR or export a CSV/Excel version."
        )

    # De-duplicate table names so that selection in the UI is unambiguous.
    seen: dict[str, int] = {}
    for table in tables:
        base = sanitize_filename(table.name, fallback="table")
        count = seen.get(base, 0) + 1
        seen[base] = count
        table.name = base if count == 1 else f"{base} ({count})"
        if table.dataframe.shape[1] == 0:
            raise FileLoadError(f"The table '{table.name}' has no columns.")
        validate_dataframe_limits(table.rows, table.columns, table_name=table.name)
        if table.rows == 0:
            raise FileLoadError(f"The table '{table.name}' contains headers but no data rows.")

    filename_safe = sanitize_filename(validation.filename)
    return LoadedFile(
        filename=filename_safe,
        file_type=validation.file_type,
        size_bytes=validation.size_bytes,
        original_bytes=data,
        tables=tables,
        warnings=warnings,
        encoding=encoding,
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def _decode_text(data: bytes) -> tuple[str, str, list[str]]:
    """Decode bytes using a short, documented fallback chain."""

    warnings: list[str] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16", warnings
        except UnicodeDecodeError:
            pass

    for encoding in _SUPPORTED_ENCODINGS:
        try:
            text = data.decode(encoding)
            if encoding != "utf-8-sig":
                warnings.append(
                    f"The file was decoded as {encoding} because it is not valid UTF-8. "
                    "Check accented characters and symbol columns."
                )
            return text, encoding if encoding != "utf-8-sig" else "utf-8", warnings
        except UnicodeDecodeError:
            continue
    raise FileLoadError(
        "CleanFlow could not decode this text file. Save it as UTF-8 and try again."
    )


def _sniff_delimiter(text: str) -> str:
    sample = text[:65_536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Fall back to the delimiter that appears most in the first line.
        first_line = next((line for line in sample.splitlines() if line.strip()), "")
        counts = {delimiter: first_line.count(delimiter) for delimiter in ",;\t|"}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _looks_headerless(columns: list[str]) -> bool:
    if not columns:
        return True
    for column in columns:
        name = str(column).strip()
        if not name:
            return True
        if name.lower().startswith("unnamed:"):
            continue
        if _NUMBERISH_RE.match(name):
            return True
        if re.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$", name):
            return True
        return False
    return True


def _load_csv(data: bytes) -> tuple[TableData, str, list[str]]:
    text, encoding, warnings = _decode_text(data)
    delimiter = _sniff_delimiter(text)
    if delimiter != ",":
        warnings.append(f"Detected a '{delimiter}' delimiter and parsed the file accordingly.")

    def read(header: int | None) -> pd.DataFrame:
        return pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            header=header,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            engine="python",
            skip_blank_lines=True,
            on_bad_lines="error",
        )

    try:
        frame = read(0)
    except pd.errors.ParserError as exc:
        raise FileLoadError(
            "The CSV structure could not be parsed. Check for inconsistent quoting, "
            "a wrong delimiter, or rows with too many columns."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise FileLoadError("The CSV file contains no readable rows.") from exc

    if _looks_headerless([str(column) for column in frame.columns]):
        try:
            frame = read(None)
            frame.columns = [f"column_{index + 1}" for index in range(frame.shape[1])]
            warnings.append(
                "No header row was detected, so columns were named column_1, column_2, … "
                "Rename them on the cleaning screen if needed."
            )
        except Exception:  # keep the first parse if the retry fails
            pass

    # pandas names a missing trailing header "Unnamed: N". Keep it when it
    # contains data; only drop it when the whole column is blank.
    unnamed = [
        column for column in frame.columns if str(column).startswith("Unnamed:")
    ]
    for index, column in enumerate(unnamed):
        if frame[column].astype(str).str.strip().eq("").all():
            frame = frame.drop(columns=[column])
        else:
            frame = frame.rename(columns={column: f"column_extra_{index + 1}"})
            warnings.append(
                "A CSV column had no header but contained data; it was named column_extra_"
                f"{index + 1}."
            )
    if frame.shape[1] == 0:
        raise FileLoadError("The CSV file has no readable columns.")
    if frame.shape[0] == 0:
        raise FileLoadError("The CSV file has headers but no data rows.")

    frame = _normalise_empty_strings(frame)
    return TableData(name="CSV data", dataframe=frame, source_type="csv"), encoding, warnings


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def _load_excel(data: bytes, file_type: str) -> tuple[list[TableData], list[str]]:
    warnings: list[str] = []
    buffer = io.BytesIO(data)
    engine = "xlrd" if file_type == "xls" else "openpyxl"

    try:
        excel = pd.ExcelFile(buffer, engine=engine)
    except ImportError as exc:
        if file_type == "xls":
            raise FileLoadError(
                "Legacy .xls files need the 'xlrd' package. Install requirements.txt, "
                "or re-save the workbook as .xlsx in Excel."
            ) from exc
        raise
    except Exception as exc:
        LOGGER.info("Excel open failed: %s", exc)
        raise FileLoadError(
            "This workbook could not be opened. It may be corrupted, encrypted, "
            "or in a format the Excel reader does not support."
        ) from exc

    try:
        sheet_names = list(excel.sheet_names)
        if len(sheet_names) > MAX_EXCEL_SHEETS:
            raise FileLoadError(
                f"This workbook has {len(sheet_names)} worksheets; the limit is {MAX_EXCEL_SHEETS}. "
                "Remove unused sheets and upload again."
            )

        tables: list[TableData] = []
        empty_sheets: list[str] = []
        for sheet_name in sheet_names:
            try:
                frame = pd.read_excel(
                    excel,
                    sheet_name=sheet_name,
                    dtype=object,
                    keep_default_na=False,
                    na_filter=False,
                )
            except Exception as exc:
                LOGGER.info("Skipping unreadable sheet %s: %s", sheet_name, exc)
                warnings.append(f"The worksheet '{sheet_name}' could not be read and was skipped.")
                continue

            if frame.shape[0] == 0 or frame.shape[1] == 0:
                empty_sheets.append(sheet_name)
                continue

            # Drop fully blank rows/columns that Excel often leaves behind.
            frame = _drop_blank_edges(frame)
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                empty_sheets.append(sheet_name)
                continue

            if _looks_headerless([str(column) for column in frame.columns]):
                # Re-read with no header and generate names rather than silently
                # treating the first data row as a header.
                try:
                    raw = pd.read_excel(
                        excel,
                        sheet_name=sheet_name,
                        header=None,
                        dtype=object,
                        keep_default_na=False,
                        na_filter=False,
                    )
                    raw = _drop_blank_edges(raw)
                    raw.columns = [f"column_{index + 1}" for index in range(raw.shape[1])]
                    frame = raw
                    warnings.append(
                        f"Worksheet '{sheet_name}' did not appear to have a header row; "
                        "columns were named automatically."
                    )
                except Exception:
                    pass

            frame = frame.reset_index(drop=True)
            if frame.shape[0] == 0:
                empty_sheets.append(sheet_name)
                continue

            tables.append(
                TableData(
                    name=sheet_name,
                    dataframe=_normalise_empty_strings(frame),
                    source_type="excel_sheet",
                )
            )

        if empty_sheets:
            warnings.append(
                "Skipped empty or unreadable worksheet(s): " + ", ".join(empty_sheets[:10])
                + ("…" if len(empty_sheets) > 10 else "")
            )
        if not tables:
            raise FileLoadError("This workbook contains no worksheets with data.")
        return tables, warnings
    finally:
        excel.close()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _load_json(data: bytes) -> tuple[list[TableData], str, list[str]]:
    text, encoding, warnings = _decode_text(data)
    if text.lstrip().startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    try:
        payload = json.loads(text)
        json_lines = False
    except json.JSONDecodeError as exc:
        json_lines, payload = _try_json_lines(text)
        if not json_lines:
            raise FileLoadError(
                f"The JSON could not be parsed: {exc.msg} at line {exc.lineno}, column {exc.colno}. "
                "Fix the syntax and upload again."
            ) from exc
        warnings.append("The file was parsed as JSON Lines (one JSON object per line).")

    tables = _json_payload_to_tables(payload)
    if not tables:
        raise FileLoadError(
            "The JSON does not contain tabular records. "
            "CleanFlow supports arrays of objects, objects containing arrays, and JSON Lines."
        )
    return tables, encoding, warnings


def _try_json_lines(text: str) -> tuple[bool, Any]:
    records: list[Any] = []
    for line in text.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            return False, None
    if not records:
        return False, None
    return True, records


def _json_payload_to_tables(payload: Any) -> list[TableData]:
    warnings: list[str] = []

    def frame_from_records(records: Any, name: str) -> TableData | None:
        try:
            if isinstance(records, list):
                if not records:
                    return None
                if all(isinstance(item, dict) for item in records):
                    frame = pd.json_normalize(records, sep="_", max_level=5)
                elif all(not isinstance(item, (dict, list)) for item in records):
                    frame = pd.DataFrame({"value": records})
                else:
                    frame = pd.DataFrame(records)
            elif isinstance(records, dict):
                frame = pd.json_normalize([records], sep="_", max_level=5)
            else:
                return None
        except Exception:
            return None

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return None
        frame = _stringify_nested_cells(frame)
        return TableData(
            name=name,
            dataframe=_normalise_empty_strings(frame.reset_index(drop=True)),
            source_type="json",
            warnings=list(warnings),
        )

    tables: list[TableData] = []

    if isinstance(payload, list):
        table = frame_from_records(payload, "JSON records")
        return [table] if table else []

    if isinstance(payload, dict):
        # Prefer separate tables for each top-level list of records. This is
        # common for API exports such as {"orders": [...], "customers": [...]}.
        list_keys = [key for key, value in payload.items() if isinstance(value, list) and value]
        if list_keys:
            for key in list_keys:
                table = frame_from_records(payload[key], str(key))
                if table:
                    tables.append(table)
            if not tables:
                return []
            if len(payload.keys()) > len(list_keys):
                warnings.append(
                    "Top-level JSON fields that were not arrays of records are not included in the tables."
                )
            for table in tables:
                table.warnings = list(warnings)
            return tables

        # A plain object of scalar fields becomes one row; nested objects are
        # flattened into column names.
        table = frame_from_records(payload, "JSON record")
        if table:
            warnings.append(
                "Nested JSON objects were flattened into underscore-separated column names."
            )
            table.warnings = list(warnings)
            return [table]

    return []


def _stringify_nested_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep arrays/objects inside cells as JSON text so every cell is scalar."""

    frame = frame.copy()
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object:
            frame[column] = series.map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
    return frame


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _load_pdf(data: bytes, *, attempt_ocr: bool) -> tuple[list[TableData], list[str], bool]:
    warnings = scan_pdf_bytes(data)
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - requirements pin it
        raise FileLoadError("PDF support requires the 'pdfplumber' package. Install requirements.txt.") from exc

    tables: list[TableData] = []
    text_pages: list[str] = []
    scanned_pages: list[int] = []

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page_count = len(pdf.pages)
            if page_count > MAX_PDF_PAGES:
                raise FileLoadError(
                    f"This PDF has {page_count} pages; the limit is {MAX_PDF_PAGES}. "
                    "Split the document and upload the relevant pages."
                )
            if page_count == 0:
                raise FileLoadError("This PDF has no pages.")

            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    page_tables = page.extract_tables() or []
                except Exception:
                    page_tables = []
                extracted_for_page = False
                for table_index, raw_table in enumerate(page_tables, start=1):
                    table = _pdf_grid_to_table(raw_table, page_number, table_index)
                    if table is not None:
                        tables.append(table)
                        extracted_for_page = True

                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    text_pages.append(page_text)
                elif not extracted_for_page:
                    scanned_pages.append(page_number)

            if not tables and text_pages:
                text = "\n".join(text_pages)
                lines = [line.rstrip() for line in text.splitlines() if line.strip()]
                if lines:
                    frame = pd.DataFrame({"text": lines})
                    tables.append(
                        TableData(
                            name="PDF text",
                            dataframe=frame,
                            source_type="pdf_text",
                            warnings=[
                                "No bordered tables were detected, so each extracted text line "
                                "was placed in a single 'text' column. You may need to split it manually."
                            ],
                        )
                    )
                    warnings.append(
                        "No structured tables were found in this PDF; text was extracted line by line."
                    )

            if not tables:
                if scanned_pages:
                    if attempt_ocr:
                        ocr_table, ocr_warnings = _ocr_pdf(data)
                        warnings.extend(ocr_warnings)
                        return [ocr_table], warnings, True
                    raise FileLoadError(
                        "This PDF appears to be scanned: no text layer or table structure was found. "
                        "OCR is required to read it. Enable 'Try OCR' on the upload screen after installing "
                        "the optional Tesseract + Poppler tools described in the README, or upload a digital PDF."
                    )
                raise FileLoadError(
                    "No readable text or tables were found in this PDF. It may be image-only or protected."
                )

            if scanned_pages and tables:
                warnings.append(
                    "Some PDF pages contained no extractable text (page "
                    + ", ".join(map(str, scanned_pages[:10]))
                    + "). Those pages may be scanned images and are not included."
                )
            return tables, warnings, False
    except FileLoadError:
        raise
    except Exception as exc:
        LOGGER.info("PDF extraction failed: %s", exc)
        raise FileLoadError(
            "This PDF could not be read. It may be corrupted, password-protected, "
            "or use a structure pdfplumber cannot parse."
        ) from exc


def _pdf_grid_to_table(raw_table: list[list[Any]], page: int, index: int) -> TableData | None:
    rows = [
        ["" if cell is None else str(cell).strip() for cell in row]
        for row in raw_table
        if row and any(str(cell).strip() for cell in row if cell is not None)
    ]
    if len(rows) < 1:
        return None

    width = max(len(row) for row in rows)
    if width == 0:
        return None
    rows = [row + [""] * (width - len(row)) for row in rows]

    # Drop columns that are empty on every row.
    keep = [col for col in range(width) if any(row[col] for row in rows)]
    if not keep:
        return None
    rows = [[row[col] for col in keep] for row in rows]

    first = rows[0]
    has_header = (
        len(rows) > 1
        and all(first)
        and len(set(first)) == len(first)
        and not any(_NUMBERISH_RE.match(value) for value in first)
        and len(first) >= 2
    )

    if has_header:
        columns = [value or f"column_{i + 1}" for i, value in enumerate(first)]
        data_rows = rows[1:]
        header_used = True
    else:
        columns = [f"column_{i + 1}" for i in range(len(first))]
        data_rows = rows
        header_used = False

    if not data_rows:
        return None
    frame = pd.DataFrame(data_rows, columns=columns)
    frame = _normalise_empty_strings(frame)
    return TableData(
        name=f"Page {page} table {index}",
        dataframe=frame,
        source_type="pdf_table",
        page=page,
        warnings=[] if header_used else ["The first row of this PDF table did not look like a header; columns were named automatically."],
    )


def _ocr_pdf(data: bytes) -> tuple[TableData, list[str]]:
    """Optional local OCR pathway.

    Requires: pytesseract, pdf2image, the Tesseract binary, and Poppler. These
    are deliberately not in requirements.txt because most users never need
    them and their system installers differ per platform.
    """

    warnings: list[str] = []
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError as exc:
        raise FileLoadError(
            "OCR was requested but the optional OCR libraries are not installed. "
            "Run: pip install -r requirements-ocr.txt, then install Tesseract OCR and Poppler, "
            "and restart the app. See the README for Windows steps."
        ) from exc

    try:
        pages = convert_from_bytes(data, dpi=200, fmt="png")
    except Exception as exc:
        LOGGER.info("OCR rendering failed: %s", exc)
        raise FileLoadError(
            "OCR could not render this PDF. Check that Poppler is installed and available on PATH."
        ) from exc

    if len(pages) > MAX_PDF_PAGES:
        raise FileLoadError(f"OCR is limited to {MAX_PDF_PAGES} pages per upload.")

    lines: list[str] = []
    for page in pages:
        try:
            text = pytesseract.image_to_string(page)
        except Exception as exc:
            LOGGER.info("OCR failed: %s", exc)
            raise FileLoadError(
                "Tesseract OCR failed. Check that the tesseract executable is installed and on PATH."
            ) from exc
        lines.extend([line.rstrip() for line in text.splitlines() if line.strip()])

    if not lines:
        raise FileLoadError("OCR ran but did not recognise any text in this PDF.")
    warnings.append("OCR output is line-based text, not a reconstructed table. Review it carefully.")
    return (
        TableData(
            name="OCR text",
            dataframe=pd.DataFrame({"text": lines}),
            source_type="pdf_text",
            warnings=list(warnings),
        ),
        warnings,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _normalise_empty_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert whitespace-only strings to NA without changing other values."""

    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: None if isinstance(value, str) and not value.strip() else value
            )
    return frame


def _is_blank_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _drop_blank_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove fully blank rows at the top/bottom and fully blank columns.

    Interior blank rows are left in place so the cleaning report can count and
    document their removal rather than losing them silently during loading.
    """

    if frame.empty:
        return frame
    frame = frame.copy()
    blank_row = frame.apply(lambda row: all(_is_blank_cell(value) for value in row), axis=1)
    blank_col = frame.apply(lambda col: all(_is_blank_cell(value) for value in col), axis=0)
    frame = frame.loc[:, ~blank_col]

    if blank_row.any() and (blank_row.iloc[0] or blank_row.iloc[-1]):
        keep_positions = [position for position, blank in enumerate(blank_row.tolist()) if not blank]
        if keep_positions:
            frame = frame.iloc[keep_positions[0] : keep_positions[-1] + 1]
    return frame.reset_index(drop=True)
