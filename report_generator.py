"""Excel and CSV exports plus the data-quality report.

Spreadsheet safety notes:

* All text cells written to Excel are forced to ``data_type = "s"``. openpyxl
  would otherwise interpret a cell beginning with "=" as a live formula.
* Excel exports force text cells to literal text, so a value beginning with
  ``=`` is not written as a formula. CSV exports for spreadsheets prefix
  formula-looking text (``=``, ``+``, ``-``, ``@``) with an apostrophe so the
  download cannot execute a formula when opened. The Power BI package opts out
  explicitly because its CSVs are import-only.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from modules.data_cleaner import CleaningResult
from modules.data_profiler import ColumnKind, ColumnProfile
from utils.security import is_formula_like, neutralise_csv_value
from utils.validators import safe_sheet_name

APP_VERSION = "1.0.0"
NAVY = "0B1F3A"
ROYAL = "1D4ED8"
CYAN = "0891B2"
LIGHT = "F2F6FC"
MID_GREY = "64748B"
WHITE = "FFFFFF"

HEADER_FILL = PatternFill("solid", fgColor=NAVY)
SECTION_FILL = PatternFill("solid", fgColor=LIGHT)
LABEL_FONT = Font(name="Calibri", bold=True, color=NAVY, size=11)
BODY_FONT = Font(name="Calibri", color="1E293B", size=10)
SMALL_GREY = Font(name="Calibri", color=MID_GREY, size=9)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
def build_cleaned_workbook(
    results: dict[str, CleaningResult],
    *,
    source_name: str,
) -> bytes:
    """Create the cleaned Excel workbook (summary + one sheet per table + log)."""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        workbook = writer.book
        workbook.properties.creator = "CleanFlow AI"
        workbook.properties.title = "CleanFlow AI cleaned data"
        workbook.properties.description = "Cleaned data generated locally by CleanFlow AI."

        summary_sheet = _first_worksheet(workbook, "Summary")
        _build_summary_sheet(
            summary_sheet,
            results=results,
            source_name=source_name,
        )

        used_sheets = {"Summary"}
        used_tables: set[str] = set()
        for table_name, result in results.items():
            sheet_name = _unique_sheet_name(safe_sheet_name(table_name, "Data"), used_sheets)
            used_sheets.add(sheet_name)
            frame = _prepare_for_excel(result.cleaned)
            if frame.shape[1] == 0:
                _write_message_sheet(
                    workbook,
                    sheet_name,
                    "No columns remain after cleaning. Review the cleaning report.",
                )
                continue
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_data_sheet(
                writer.sheets[sheet_name],
                frame,
                result.profile.column_profiles,
                table_name=sheet_name,
                used_table_names=used_tables,
            )

        # Combined audit log.
        log_rows: list[dict[str, Any]] = []
        for table_name, result in results.items():
            for entry in result.audit:
                row = entry.to_dict()
                row = {"table": table_name, **row}
                log_rows.append(row)
        log_frame = pd.DataFrame(log_rows)
        if log_frame.empty:
            log_frame = pd.DataFrame(
                [{"table": "—", "message": "No transformations were required with the selected options."}]
            )
            log_sheet = _write_plain_frame(writer, log_frame, "Cleaning Log")
        else:
            log_sheet = _write_plain_frame(writer, log_frame, "Cleaning Log")
        _write_note_row(
            writer.sheets[log_sheet],
            len(log_frame) + 3,
            "Audit log. The original uploaded file is unchanged and available separately.",
        )

    return buffer.getvalue()


def build_cleaned_csv(result: CleaningResult) -> bytes:
    """Export one cleaned table as UTF-8 CSV, neutralised for spreadsheets."""

    return dataframe_to_csv(result.cleaned, neutralise=True)


def dataframe_to_csv(frame: pd.DataFrame, *, neutralise: bool = False) -> bytes:
    """Serialise a DataFrame to UTF-8-BOM CSV with ISO dates."""

    clean = _prepare_for_csv(frame)
    if neutralise:
        for position in range(clean.shape[1]):
            series = clean.iloc[:, position]
            if series.dtype == object:
                clean.isetitem(position, series.map(neutralise_csv_value))
    csv_text = clean.to_csv(index=False, encoding="utf-8", lineterminator="\n")
    return ("\ufeff" + csv_text).encode("utf-8")


def build_quality_report(result: CleaningResult, *, source_name: str) -> bytes:
    """Create the multi-sheet data-quality report."""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        workbook = writer.book
        workbook.properties.creator = "CleanFlow AI"
        workbook.properties.title = "CleanFlow AI data quality report"

        overview = _first_worksheet(workbook, "Overview")
        _build_quality_overview(overview, result, source_name)

        profile_frame = result.quality_frame()
        if not profile_frame.empty:
            friendly = profile_frame.rename(
                columns={
                    "column": "Column",
                    "type": "Detected type",
                    "role": "Analytics role",
                    "dtype": "Pandas dtype",
                    "non_null": "Non-null",
                    "missing": "Missing",
                    "missing_pct": "Missing %",
                    "unique": "Unique",
                    "minimum": "Minimum",
                    "maximum": "Maximum",
                    "mean": "Mean",
                    "median": "Median",
                    "total": "Total",
                    "issues": "Issues",
                    "sample": "Example values",
                    "currency": "Currency",
                    "outliers": "Outliers",
                    "date_format": "Date format",
                    "ambiguous_dates": "Ambiguous dates",
                }
            )
            _write_plain_frame(writer, friendly, "Column Profile")
            _format_profile_sheet(writer.sheets["Column Profile"], result.profile.column_profiles)
        else:
            _write_message_sheet(workbook, "Column Profile", "No columns were available to profile.")

        conversions = pd.DataFrame(result.conversions)
        if conversions.empty:
            _write_message_sheet(workbook, "Type Conversions", "No type conversions were applied.")
        else:
            conversions = conversions.rename(
                columns={
                    "column": "Column",
                    "from": "From",
                    "to": "To",
                    "values_converted": "Values converted",
                    "values_flagged": "Values flagged",
                    "notes": "Notes",
                    "examples": "Flagged examples",
                }
            )
            _write_plain_frame(writer, conversions, "Type Conversions")

        invalid = pd.DataFrame(result.invalid_values)
        if invalid.empty:
            _write_message_sheet(
                workbook,
                "Invalid Values",
                "No invalid values were found within the selected conversion threshold. "
                "Empty source cells are not counted as invalid.",
            )
        else:
            invalid = invalid.rename(
                columns={"column": "Column", "row": "Spreadsheet row", "value": "Original value", "problem": "Problem"}
            )
            _write_plain_frame(writer, invalid, "Invalid Values")

        duplicates = result.duplicates_removed_frame
        if duplicates.empty:
            _write_message_sheet(
                workbook,
                "Duplicates",
                "No exact duplicate rows were removed. Duplicate detection uses all columns, never a single ID.",
            )
        else:
            _write_plain_frame(writer, _prepare_for_excel(duplicates), "Duplicates")
            _write_note_row(
                writer.sheets["Duplicates"],
                len(duplicates) + 3,
                "Showing at most 2,000 removed rows. Duplicate count in the Overview is complete.",
            )

        category_rows: list[dict[str, Any]] = []
        for item in result.category_groups:
            for group in item["groups"]:
                for variant in group["variants"]:
                    category_rows.append(
                        {
                            "Column": item["column"],
                            "Normalised key": group["normalized"],
                            "Variant": variant["value"],
                            "Count": variant["count"],
                            "Suggested canonical": group["canonical"],
                            "Group total": group["total"],
                        }
                    )
        if category_rows:
            _write_plain_frame(writer, pd.DataFrame(category_rows), "Categories")
            _write_note_row(
                writer.sheets["Categories"],
                len(category_rows) + 3,
                "These labels differ only by letter case. They were reported, and merged only if "
                "category case normalization was enabled.",
            )
        else:
            _write_message_sheet(
                workbook,
                "Categories",
                "No category labels differed only by letter case.",
            )

        outliers = result.outliers_frame()
        if outliers.empty:
            _write_message_sheet(
                workbook,
                "Outliers",
                "No IQR outliers were flagged (or outlier flagging was disabled). "
                "Outliers are never deleted automatically.",
            )
        else:
            _write_plain_frame(writer, outliers, "Outliers")
            _write_note_row(
                writer.sheets["Outliers"],
                len(outliers) + 3,
                "Outliers are shown for review only. No outlier rows were removed.",
            )

        audit = result.audit_frame()
        if audit.empty:
            _write_message_sheet(workbook, "Audit Log", "No transformations were required.")
        else:
            _write_plain_frame(writer, audit, "Audit Log")
            _write_note_row(
                writer.sheets["Audit Log"],
                len(audit) + 3,
                "Every operation is recorded with its step, scope, and affected counts.",
            )

    return buffer.getvalue()


def build_activity_log_csv(result: CleaningResult) -> bytes:
    frame = result.audit_frame()
    if frame.empty:
        frame = pd.DataFrame(
            [{"message": "No transformations were required with the selected options."}]
        )
    return dataframe_to_csv(frame, neutralise=True)


def build_quality_summary_csv(result: CleaningResult) -> bytes:
    return dataframe_to_csv(result.quality_frame(), neutralise=True)


# ---------------------------------------------------------------------------
# Excel internals
# ---------------------------------------------------------------------------
def _first_worksheet(workbook: Workbook, title: str) -> Any:
    """Return the workbook's first sheet, creating or renaming as needed.

    pandas' openpyxl writer can leave ``workbook.active`` as ``None``; never
    assume a default sheet exists.
    """

    if workbook.sheetnames:
        sheet = workbook[workbook.sheetnames[0]]
        sheet.title = _unique_sheet_name(safe_sheet_name(title, "Sheet"), set(workbook.sheetnames[1:]))
    else:
        sheet = workbook.create_sheet(title)
    return sheet


def _build_summary_sheet(sheet: Any, *, results: dict[str, CleaningResult], source_name: str) -> None:
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 30
    sheet.column_dimensions["B"].width = 62
    sheet.column_dimensions["C"].width = 22
    sheet.column_dimensions["D"].width = 22

    sheet.merge_cells("A1:D1")
    title = sheet["A1"]
    title.value = "CleanFlow AI"
    title.font = Font(name="Calibri", bold=True, size=22, color=WHITE)
    title.fill = HEADER_FILL
    title.alignment = Alignment(vertical="center", horizontal="left", indent=1)
    sheet.row_dimensions[1].height = 44

    sheet.merge_cells("A2:D2")
    subtitle = sheet["A2"]
    subtitle.value = "Cleaned data export"
    subtitle.font = Font(name="Calibri", bold=True, size=13, color=ROYAL)
    subtitle.alignment = Alignment(vertical="center", indent=1)
    sheet.row_dimensions[2].height = 26

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows_before = sum(result.summary.rows_before for result in results.values())
    rows_after = sum(result.summary.rows_after for result in results.values())
    duplicates = sum(result.summary.duplicates_removed for result in results.values())
    invalid = sum(result.summary.invalid_values for result in results.values())

    row = 4
    row = _section(sheet, row, "Export details")
    for label, value in (
        ("Source file", source_name),
        ("Generated (UTC)", generated),
        ("Tables in this workbook", ", ".join(results.keys())),
        ("Rows before cleaning", f"{rows_before:,}"),
        ("Rows after cleaning", f"{rows_after:,}"),
        ("Exact duplicates removed", f"{duplicates:,}"),
        ("Invalid converted values", f"{invalid:,}"),
    ):
        row = _label_value(sheet, row, label, value)

    row += 1
    row = _section(sheet, row, "Table summary")
    header_row = row
    for index, heading in enumerate(("Table", "Rows", "Columns", "Duplicates removed"), start=1):
        cell = sheet.cell(row=header_row, column=index, value=heading)
        cell.font = Font(bold=True, color=WHITE, size=10)
        cell.fill = PatternFill("solid", fgColor=ROYAL)
        cell.alignment = Alignment(vertical="center")
    row += 1
    for table_name, result in results.items():
        sheet.cell(row=row, column=1, value=table_name).font = BODY_FONT
        sheet.cell(row=row, column=2, value=result.summary.rows_after).font = BODY_FONT
        sheet.cell(row=row, column=3, value=result.summary.columns_after).font = BODY_FONT
        sheet.cell(row=row, column=4, value=result.summary.duplicates_removed).font = BODY_FONT
        row += 1

    row += 1
    row = _section(sheet, row, "How to use this workbook")
    notes = [
        "Each source table is on its own sheet with an Excel table, filters, frozen header, and sensible formats.",
        "The original uploaded file was not modified. Download it separately from CleanFlow AI for comparison.",
        "Missing values are left blank. No values were imputed, filled, or invented.",
        "Identifiers such as IDs, phone numbers, and postal codes are stored as text to preserve leading zeros.",
        "Percentages are stored as percentage points: 12.5% is written as 12.5, not 0.125.",
        "Every transformation is listed on the Cleaning Log sheet and in the data-quality report.",
    ]
    for note in notes:
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        cell = sheet.cell(row=row, column=1, value="•  " + note)
        cell.font = BODY_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row].height = 30
        row += 1

    if any(result.warnings for result in results.values()):
        row += 1
        row = _section(sheet, row, "Important notes")
        for result in results.values():
            for warning in result.warnings:
                sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
                cell = sheet.cell(row=row, column=1, value="•  " + warning)
                cell.font = SMALL_GREY
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                sheet.row_dimensions[row].height = 28
                row += 1

    sheet.freeze_panes = "A4"
    sheet.sheet_properties.tabColor = ROYAL


def _build_quality_overview(sheet: Any, result: CleaningResult, source_name: str) -> None:
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 76

    sheet.merge_cells("A1:B1")
    title = sheet["A1"]
    title.value = "CleanFlow AI — Data Quality Report"
    title.font = Font(bold=True, size=20, color=WHITE)
    title.fill = HEADER_FILL
    title.alignment = Alignment(vertical="center", indent=1)
    sheet.row_dimensions[1].height = 42

    sheet.merge_cells("A2:B2")
    subtitle = sheet["A2"]
    subtitle.value = f"Table: {result.table_name}   |   Source: {source_name}"
    subtitle.font = Font(size=11, color=ROYAL, bold=True)
    subtitle.alignment = Alignment(indent=1)
    sheet.row_dimensions[2].height = 24

    row = 4
    row = _section(sheet, row, "Quality summary")
    summary = result.summary
    metrics = [
        ("Rows before cleaning", f"{summary.rows_before:,}"),
        ("Rows after cleaning", f"{summary.rows_after:,}"),
        ("Columns before cleaning", f"{summary.columns_before:,}"),
        ("Columns after cleaning", f"{summary.columns_after:,}"),
        ("Missing cells before", f"{summary.missing_cells_before:,}"),
        ("Missing cells after", f"{summary.missing_cells_after:,}"),
        ("Blank placeholder cells normalised", f"{summary.missing_tokens_replaced:,}"),
        ("Exact duplicate rows removed", f"{summary.duplicates_removed:,}"),
        ("Empty rows detected / removed", f"{summary.empty_rows_found:,} / {summary.empty_rows_removed:,}"),
        ("Empty columns detected / removed", f"{summary.empty_columns_found:,} / {summary.empty_columns_removed:,}"),
        ("Cells trimmed", f"{summary.cells_trimmed:,}"),
        ("Cells text-normalised", f"{summary.cells_text_normalized:,}"),
        ("Category labels case-normalised", f"{summary.categories_normalized:,}"),
        ("Columns renamed", f"{summary.columns_renamed:,}"),
        ("Valid type conversions", f"{summary.values_converted:,}"),
        ("Invalid conversion values", f"{summary.invalid_values:,}"),
        ("Possible outliers flagged", f"{summary.outliers_flagged:,}"),
        ("Cleaning duration", f"{summary.duration_ms / 1000:.2f} s"),
        ("Report generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
        ("CleanFlow AI version", APP_VERSION),
    ]
    for label, value in metrics:
        row = _label_value(sheet, row, label, value)

    row += 1
    row = _section(sheet, row, "Cleaning options used")
    for label, value in result.options.to_dict().items():
        row = _label_value(sheet, row, label, value)

    if result.column_renames:
        row += 1
        row = _section(sheet, row, "Column names")
        for item in result.column_renames:
            row = _label_value(sheet, row, item["before"], item["after"])

    if result.warnings:
        row += 1
        row = _section(sheet, row, "Important notes")
        for warning in result.warnings:
            sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
            cell = sheet.cell(row=row, column=1, value="•  " + warning)
            cell.font = BODY_FONT
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.row_dimensions[row].height = 32
            row += 1

    row += 1
    row = _section(sheet, row, "Report contents")
    for sheet_name, description in (
        ("Column Profile", "Type, role, missingness, statistics, examples, and issues for every column."),
        ("Type Conversions", "What changed, how many values converted, and what was flagged."),
        ("Invalid Values", "Original values that could not be converted, with spreadsheet row numbers."),
        ("Duplicates", "Rows removed as exact duplicates across all columns."),
        ("Categories", "Labels that differ only by letter case, with counts and a suggested canonical spelling."),
        ("Outliers", "IQR outlier examples for review only; nothing was removed."),
        ("Audit Log", "Chronological record of every cleaning operation."),
    ):
        row = _label_value(sheet, row, sheet_name, description)

    sheet.freeze_panes = "A4"
    sheet.sheet_properties.tabColor = ROYAL


def _section(sheet: Any, row: int, title: str) -> int:
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
    cell = sheet.cell(row=row, column=1, value=title.upper())
    cell.font = Font(bold=True, color=NAVY, size=10)
    cell.fill = SECTION_FILL
    cell.alignment = Alignment(vertical="center", indent=1)
    sheet.row_dimensions[row].height = 22
    return row + 1


def _label_value(sheet: Any, row: int, label: str, value: Any) -> int:
    label_cell = sheet.cell(row=row, column=1, value=_safe_text(label))
    label_cell.font = LABEL_FONT
    label_cell.alignment = Alignment(vertical="top", wrap_text=True)
    label_cell.fill = PatternFill("solid", fgColor="FAFBFE")

    value_cell = sheet.cell(row=row, column=2, value=_safe_cell_value(value))
    value_cell.font = BODY_FONT
    value_cell.alignment = Alignment(vertical="top", wrap_text=True)
    if isinstance(value, str) and is_formula_like(value):
        value_cell.data_type = "s"
    if isinstance(value, str) and len(value) > 90:
        sheet.row_dimensions[row].height = 32
    return row + 1


def _style_data_sheet(
    sheet: Any,
    frame: pd.DataFrame,
    profiles: list[ColumnProfile],
    *,
    table_name: str,
    used_table_names: set[str],
) -> None:
    row_count, column_count = frame.shape
    if column_count == 0:
        return

    header_fill = HEADER_FILL
    thin = Side(style="thin", color="D8E0EC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for column_index in range(1, column_count + 1):
        cell = sheet.cell(row=1, column=column_index)
        cell.font = Font(name="Calibri", bold=True, color=WHITE, size=10)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", horizontal="left", wrap_text=False)
        cell.border = border
    sheet.row_dimensions[1].height = 26
    sheet.freeze_panes = "A2"

    # Force every text cell to be literal text, never a formula. openpyxl sets
    # data_type="f" automatically for strings starting with "=".
    for row in sheet.iter_rows(min_row=2, max_row=row_count + 1, max_col=column_count):
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
                if len(cell.value) > 32767:
                    cell.value = cell.value[:32767]
                cell.alignment = Alignment(vertical="top", wrap_text=False)
            cell.border = border

    formats = {profile.name: _excel_number_format(profile) for profile in profiles}
    for column_index, column_name in enumerate(frame.columns, start=1):
        number_format = formats.get(str(column_name))
        if number_format:
            for row in range(2, row_count + 2):
                sheet.cell(row=row, column=column_index).number_format = number_format

    widths = _column_widths(frame)
    for column_index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column_index)].width = width

    if row_count > 0:
        reference = f"A1:{get_column_letter(column_count)}{row_count + 1}"
        table = Table(displayName=_unique_table_name(table_name, used_table_names), ref=reference)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        try:
            sheet.add_table(table)
        except Exception:
            # A table can fail if headers collide after export; filters still
            # provide a usable workbook.
            sheet.auto_filter.ref = reference
    else:
        sheet.auto_filter.ref = f"A1:{get_column_letter(column_count)}1"

    sheet.sheet_properties.tabColor = CYAN


def _format_profile_sheet(sheet: Any, profiles: list[ColumnProfile]) -> None:
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 26
    for cell in sheet[1]:
        cell.font = Font(bold=True, color=WHITE, size=10)
        cell.fill = HEADER_FILL
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.font = BODY_FONT
    for index, width in enumerate((22, 16, 14, 14, 10, 10, 10, 14, 14, 14, 14, 16, 50, 50, 14, 12, 20, 14), start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    if profiles:
        sheet.auto_filter.ref = f"A1:R{len(profiles) + 1}"


def _write_plain_frame(writer: pd.ExcelWriter, frame: pd.DataFrame, sheet_name: str) -> str:
    safe = _unique_sheet_name(safe_sheet_name(sheet_name), set(writer.book.sheetnames))
    prepared = _prepare_for_excel(frame)
    prepared.to_excel(writer, sheet_name=safe, index=False)
    sheet = writer.sheets[safe]
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 26
    for cell in sheet[1]:
        cell.font = Font(bold=True, color=WHITE, size=10)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.font = BODY_FONT
    widths = _column_widths(prepared)
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    if len(prepared) > 0 and prepared.shape[1] > 0:
        sheet.auto_filter.ref = f"A1:{get_column_letter(prepared.shape[1])}{len(prepared) + 1}"
    sheet.sheet_properties.tabColor = CYAN
    return safe


def _write_message_sheet(workbook: Workbook, sheet_name: str, message: str) -> None:
    safe = _unique_sheet_name(safe_sheet_name(sheet_name), set(workbook.sheetnames))
    sheet = workbook.create_sheet(safe)
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 100
    cell = sheet.cell(row=1, column=1, value=_safe_text(message))
    cell.font = BODY_FONT
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.sheet_properties.tabColor = CYAN


def _write_note_row(sheet: Any, row: int, note: str) -> None:
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=min(sheet.max_column, 6))
    cell = sheet.cell(row=row, column=1, value=_safe_text(note))
    cell.font = SMALL_GREY
    cell.alignment = Alignment(wrap_text=True, vertical="top")


def _prepare_for_excel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    prepared = frame.copy()
    # Excel tables require unique header labels. This is an export-only copy,
    # so the cleaned DataFrame and its source column names remain untouched.
    prepared.columns = _unique_export_headers(prepared.columns)
    for position in range(prepared.shape[1]):
        series = prepared.iloc[:, position]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            prepared.isetitem(position, series.dt.tz_convert("UTC").dt.tz_localize(None))
        elif series.dtype == object:
            prepared.isetitem(position, series.map(_clean_excel_value))
    return prepared


def _clean_excel_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.to_pydatetime()
    if isinstance(value, (date, datetime)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def _safe_text(text: Any) -> str:
    cleaned = ILLEGAL_CHARACTERS_RE.sub("", str(text))
    return cleaned[:32767]


def _safe_cell_value(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_text(value)
    return value


def _prepare_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for position in range(prepared.shape[1]):
        series = prepared.iloc[:, position]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            prepared.isetitem(position, series.dt.tz_convert("UTC").dt.tz_localize(None))
        elif series.dtype == object:
            prepared.isetitem(position, series.map(_clean_csv_value))
    return prepared


def _clean_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        if value.time() == datetime.min.time():
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _excel_number_format(profile: ColumnProfile) -> str | None:
    if profile.kind == ColumnKind.PERCENTAGE:
        return '0.0"%"'
    if profile.kind == ColumnKind.CURRENCY:
        symbol = profile.currency_symbols[0] if len(profile.currency_symbols) == 1 else None
        if symbol in {"₹", "$", "€", "£", "¥"}:
            return f'"{symbol}"#,##0.00'
        return "#,##0.00"
    if profile.kind == ColumnKind.DATE:
        return "yyyy-mm-dd"
    if profile.kind == ColumnKind.DATETIME:
        return "yyyy-mm-dd hh:mm"
    if profile.kind == ColumnKind.NUMERIC:
        return "#,##0" if profile.unique and all(
            isinstance(value, int) for value in [profile.minimum, profile.maximum]
        ) else "#,##0.00"
    if profile.kind == ColumnKind.IDENTIFIER:
        return "@"
    return None


def _column_widths(frame: pd.DataFrame, max_width: int = 46) -> list[float]:
    widths: list[float] = []
    for position, column in enumerate(frame.columns):
        header_length = len(str(column))
        series = frame.iloc[:200, position]
        sample_lengths = []
        for value in series:
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            sample_lengths.append(len(str(value)))
        width = max([header_length + 2, 12, *(sample_lengths[:50] or [0])]) + 2
        widths.append(min(max(width, 12), max_width))
    return widths


def _unique_export_headers(columns: Iterable[Any]) -> list[str]:
    """Make Excel-safe headers without changing the in-memory cleaned data."""

    headers: list[str] = []
    used: set[str] = set()
    for index, column in enumerate(columns, start=1):
        base = _safe_text(column).strip() or f"Column {index}"
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base} ({suffix})"
            suffix += 1
        used.add(candidate.casefold())
        headers.append(candidate)
    return headers


def _unique_sheet_name(name: str, used: set[str]) -> str:
    base = safe_sheet_name(name, "Sheet")[:31]
    candidate = base
    suffix = 2
    while candidate in used:
        tail = f" ({suffix})"
        candidate = base[: 31 - len(tail)] + tail
        suffix += 1
    return candidate


def _unique_table_name(name: str, used: set[str] | None = None) -> str:
    cleaned = "".join(character for character in str(name) if character.isalnum() or character == "_")
    cleaned = cleaned or "CleanFlowTable"
    if not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = "T_" + cleaned
    base = ("CleanFlow_" + cleaned)[:230]
    candidate = base
    suffix = 2
    if used is not None:
        while candidate in used:
            tail = f"_{suffix}"
            candidate = base[: 240 - len(tail)] + tail
            suffix += 1
        used.add(candidate)
    return candidate


def _write_safe(workbook: Workbook) -> bytes:
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
