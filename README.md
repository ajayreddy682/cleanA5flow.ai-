# CleanFlow AI

CleanFlow AI is a local-first Streamlit application for cleaning messy business files and
turning them into analysis-ready data. Upload an Excel workbook, CSV, JSON export, or
digital PDF; review the quality snapshot; choose explicit cleaning options; inspect every
transformation; then export cleaned Excel/CSV, a data-quality report, and a Power BI-ready
dataset.

No paid AI API, external service, or API key is required. Files are processed in memory for
the current session, and cleaning history stores metadata only.

## Requirements

- Python 3.10 or newer (tested on Python 3.13)
- Windows, macOS, or Linux
- Streamlit 1.49 or newer (`width="stretch"` is used throughout the UI)

## Quickstart

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\streamlit.exe run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local URL (normally <http://localhost:8501>). The app binds to
`localhost` by default; see [Deployment caveats](#deployment-caveats) before changing that.

## Workflow

1. **Upload & inspect.** Files are validated by extension, byte signature, and structure.
   Excel sheets, detected JSON arrays, and PDF tables are shown before anything changes.
2. **Clean with explicit options.** Text normalization, missing-value tokens, type/date
   conversion, duplicate removal, empty row/column removal, and outlier flagging are all
   separate controls. Defaults never invent, impute, or silently replace values.
3. **Review the audit trail.** Every operation is recorded with affected rows, cells, and
   columns. Invalid conversions are listed with their original values and spreadsheet rows.
4. **Analyse and export.** Explore Plotly KPIs and charts, filter the cleaned data, and
   download Excel, CSV, quality reports, audit logs, or a Power BI model package.

## Supported files and limits

| Format | Notes |
|---|---|
| `.xlsx` | Multiple worksheets; cached cell values only; macros and external links are never executed |
| `.xls` | Legacy binary workbooks via `xlrd` |
| `.csv` | Delimiter, header, encoding (UTF-8/CP1252) and decimal-separator detection |
| `.json` | Arrays of records, objects containing arrays, and JSON Lines |
| `.pdf` | Digital tables and text; scanned pages need optional OCR |

Limits per upload:

- 25 MB file size
- 250,000 rows, 300 columns, and 5,000,000 cells per table
- 50 worksheets per workbook
- 100 pages per PDF
- 300 MB maximum uncompressed `.xlsx` size and a 200:1 compression-ratio guard

Cleaning is intentionally conservative. Numeric and date columns are converted only when
the share of parseable values meets the selected threshold (95% by default); otherwise the
column stays text so values are not deleted. Exact duplicates must match on **all** columns.
Identifiers such as IDs, postal codes, and phone numbers remain text so leading zeros
survive. Percentages are stored as percentage points (`12.5%` becomes `12.5`, not `0.125`).
Ambiguous numeric dates are day-first by default; choose the order explicitly when the
source is month-first.

## Exports

- **Cleaned Excel workbook** — one sheet per cleaned table, plus a summary and cleaning log.
  Text cells are written as text so formula-looking values are not executed.
- **Cleaned CSV / filtered CSV / profile CSV** — UTF-8 with BOM. Formula-looking cells
  beginning with `=`, `+`, `-`, or `@` are prefixed with an apostrophe to prevent spreadsheet
  formula injection. Numeric values are untouched.
- **Data quality report** — multi-sheet `.xlsx` with the profile, conversions, invalid
  values, duplicates, categories, outliers, and audit log.
- **Power BI package** — CSVs, a documentation workbook, `model.json`, data dictionary,
  relationship suggestions, DAX measures, and an import guide. It does **not** contain a
  `.pbix` report. Package CSVs preserve raw values for import and are not neutralised for
  Excel; import them into Power BI rather than opening them in Excel.

## Privacy and local-first design

- Uploaded bytes are handled in memory for the active Streamlit session and are never
  written to a shared upload directory or served by a web server.
- Cleaning history (`.cleanflow/history.json`) contains filenames, counts, operations,
  status, and timings only. It never contains cell values or file bytes.
- No analytics, telemetry, or external API calls are made by the app. The only outbound
  request is the theme's Google Fonts import (`assets/styles.css`), which fetches the Inter
  font and never includes uploaded data. For a fully offline deployment, remove that
  `@import` line and use the system font stack.
- The history file is capped at 50 jobs and ignored by Git. Set
  `CLEANFLOW_HISTORY_PATH` to relocate it (useful on read-only installs).
- "Clear current dataset" removes the session's uploaded data and results; "Clear cleaning
  history" removes metadata entries only.

## Deployment caveats

This build is designed for **single-user local use**. It has no authentication and no
per-user data isolation:

- History is one global JSON file shared by every visitor to the process.
- Anyone who can reach the app can upload files, read history, and clear it.
- The bundled configuration binds to `localhost` on purpose. If you need remote access,
  put the app behind an authenticated reverse proxy **and** replace the file-based history
  with per-user storage before exposing it. Do not set `address = "0.0.0.0"` on an
  untrusted network.
- On ephemeral hosts (for example a free Streamlit deployment), the history file resets
  when the container is rebuilt and the filesystem may be read-only.
- The app is single-process; the history store's lock does not coordinate multiple
  Streamlit processes or replicas.

## Optional OCR for scanned PDFs

Digital PDFs work out of the box. For scanned documents, install the Python extras and the
native tools, then enable "Try OCR if this PDF is scanned" in the uploader:

```bash
python -m pip install -r requirements-ocr.txt
```

On Windows you also need the UB Mannheim Tesseract build and Poppler on `PATH`
(`tesseract --version` must work in a new terminal). On macOS, `brew install tesseract
poppler`; on Debian/Ubuntu, `sudo apt install tesseract-ocr poppler-utils`.

OCR output is line-based text rather than reconstructed tables, so review it carefully.

## Project structure

```text
app.py                     Streamlit entry point, navigation, pages, session state
modules/
  file_loader.py           Validation-aware CSV/XLSX/XLS/JSON/PDF loading
  data_profiler.py         Column typing, statistics, chart and KPI recommendations
  data_cleaner.py          Cleaning pipeline, audit log, conversion reporting
  dashboard_generator.py   Plotly charts, filters, manual chart builder
  report_generator.py      Excel/CSV exports and the data-quality report
  powerbi_exporter.py      Star-schema suggestions, DAX, and the Power BI ZIP
  history_store.py         Metadata-only local history (50 jobs)
  ui.py                    Reusable HTML/CSS components
utils/
  validators.py            File signatures, size/row limits, safe names
  security.py              Formula neutralisation, safe errors, PDF warnings
tests/                     Pytest suite, including Streamlit AppTest smoke tests
sample_data/               Messy sample files and the committed invoice PDF
assets/styles.css          Application theme
.streamlit/config.toml     Theme and server configuration
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers the cleaning engine, all loaders, Excel/CSV/report exports, the Power BI
package, dashboard date filters, and the main Streamlit flow (sample load -> clean ->
downloads -> analytics). `browser-tools/` contains development screenshot helpers and is
not part of the application.

## Troubleshooting

- **`streamlit` is not recognized** — activate the virtual environment, or run
  `.venv\Scripts\streamlit.exe run app.py` on Windows.
- **Port already in use** — `streamlit run app.py --server.port 8502`.
- **"Could not save CleanFlow history"** — the install directory is not writable; set
  `CLEANFLOW_HISTORY_PATH` to a writable location.
- **PDF has no usable text** — it is probably scanned; install the optional OCR tooling.
- **An `.xlsx` is rejected** — it may be corrupted, password-protected, or expand beyond
  the documented limits.
- **Cleaning is slow on large files** — processing is single-threaded and in memory. A
  100,000-row, 8-column file takes roughly 30 seconds with default options on a typical
  laptop; files near the 250,000-row limit can take minutes. Split very large workbooks.

## Known limitations

- There is no `.pbix` generation and no Power BI automation; the package is a data/model
  blueprint that you import into Power BI Desktop.
- The "AI" in the product name is not a hosted model; cleaning is deterministic and
  rule-based, with no paid API.
- Outliers are flagged with the IQR rule and never removed automatically.
- Missing values are never filled. Category case merging is available but off by default so
  labels are not changed silently.
- Formula neutralisation applies to CSV exports intended for spreadsheets; Power BI package
  CSVs deliberately preserve raw values for import.
- The app is not a multi-user service; see [Deployment caveats](#deployment-caveats).

## License

No license has been chosen for this project. Licensing is the owner's decision; add a
`LICENSE` file before distributing it.
