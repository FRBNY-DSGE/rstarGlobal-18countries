# Pulling and updating the data

This document covers the Python data-pull pipeline in `scripts/data/` — how to
run it, how to push its output into the MATLAB model's `indata/`, and how to
extend the sample forward by a year. See `INSTRUCTIONS.md` for how the MATLAB
side then uses the resulting workbooks.

## 1. What the master pull does

`scripts/data/master-pull.py` rebuilds the two workbooks the MATLAB models read
(`DataInflShortLongConsUpdated.xlsx` and its no-consumption sibling
`DataInflShortLongUpdated.xlsx`) by starting from the existing workbook in
`indata/` and refreshing only the most recent years, country by country,
straight from each series' original source.

Note on layout: unlike some other copies of this pipeline, this checkout is
flat — everything the pull needs (`master-pull.py`, `api_keys.py`, this
`DATA.md`) lives directly in `scripts/data/`. There is no separate
`data/master/`, `data/aggregate/`, or per-country `data/<country>/` notebook
tree; the script reads/writes straight against `indata/` (see the `INDATA`
constant, `os.path.join(HERE, "..", "..", "indata")`, near the top of
`master-pull.py`).

Rule: for every column, years **≤ `ANCHOR` (2020)** are kept exactly as they are
in the source workbook. Years **`FETCH_START`–`FETCH_END` (2021–2025 by
default)** are replaced with freshly fetched values. `cpi_*` and `rconpc_*` are
index series (rebased so a base year = 100); if the freshly fetched source level
at the anchor year differs from the existing workbook by more than
`CHAIN_REANCHOR_PCT` (0.5%), the script pins the anchor year to the existing
value and chains the new years forward from there using the source's
year-over-year growth rates, instead of splicing in a discontinuous level. Rate
columns (`stir_*`, `ltir_*`, `baa_usa`) are placed directly, no chaining.

Outputs, written directly into `indata/`:

- `DataInflShortLongConsUpdated_2025.xlsx` — full workbook (with `rconpc_*`)
- `DataInflShortLongUpdated_2025.xlsx` — same data, `rconpc_*` columns stripped

A fetch that fails, or a fetch that returns nothing for the target window, is
logged (`[FAIL ]` / `[warn ]`) and that column's existing values are left
untouched rather than blanked out — check the console output after a run for
any column that didn't refresh.

### Required Python packages

```bash
pip install pandas openpyxl requests playwright
playwright install chromium
```

- `pandas`, `openpyxl` — reading/writing the Excel workbooks.
- `requests` — fetching from FRED, OECD, IMF, World Bank, and the other
  HTTP/CSV sources used by the per-country fetch functions.
- `playwright` (+ its Chromium install) — needed specifically for the UK long
  rate (`ltir_uk_boe`), since the Bank of England site requires a headless
  browser to download its CSV export. If you don't need to refresh `ltir_uk`,
  you can skip this and the rest of the script still runs (that one fetch
  will fail and fall back to existing values).

Also required: `scripts/data/api_keys.py` (a local, restricted-permission file
defining `FRED_KEY` and `BDF_KEY`, imported via `from api_keys import
BDF_KEY, FRED_KEY`). This file is not meant to be committed or shared — if
you're setting up a fresh checkout and it's missing, you'll need to recreate
it with valid API keys before running the pull.

### Australia short rate — manual step required

`stir_au` (Australia's short-term rate) is **not fetched automatically** like
every other series. `sr_australia()` in `master-pull.py` just reads a local
file, `indata/raw/f1.1-data.csv`, and does not download anything itself. If
that file is stale, `stir_au` will silently stop updating past whatever year
is in it while every other column keeps refreshing normally.

Before running a pull you intend to use, manually download the latest RBA
**Table F1.1 — Interest Rates and Yields, Money Market** CSV from
<https://www.rba.gov.au/statistics/tables/> and save it over
`indata/raw/f1.1-data.csv` (same format the script already expects: a header
block it skips via `skiprows=11`, then `date,value` rows). **If you (Claude)
are running this pipeline and can't fetch that file yourself, tell the user
explicitly that `indata/raw/f1.1-data.csv` needs to be refreshed by hand from
that RBA link before the pull can be trusted for the current year, rather
than silently pulling with a stale Australia short rate.**

### Running it

```bash
cd scripts/data
python master-pull.py
```

OECD endpoints are aggressively rate-limited, so every OECD-sourced measure is
fetched **once for all countries** in a single batched request and cached both
in-memory and on disk under `scripts/data/_cache/`. If you re-run the script
and it reports `(OECD ... loaded ... areas from disk cache)`, that's
expected — delete `scripts/data/_cache/` if you need to force a fresh OECD
pull (e.g. after fixing a transient failure).

### Completeness check (do this before using a pull's output)

`master-pull.py` does **not** itself refuse to write an incomplete year — if
the CPI/rate/consumption source for some column hasn't posted `FETCH_END` yet
(common: World Bank `rconpc_*` and some OECD rate series typically lag by
about a year), that column is simply left blank for that year while every
other column fills in normally, and the workbook still gets saved with that
partially-filled trailing year.

**If you are Claude running this pipeline, after every pull open the output
workbook(s) (`indata/DataInflShortLongConsUpdated_2025.xlsx` /
`indata/DataInflShortLongUpdated_2025.xlsx`) and check the last 1-2 rows
(years) for missing/blank cells across every column** — e.g. with `openpyxl`,
read the header and the row for `FETCH_END` (and `FETCH_END - 1`) and list any
column that's `None`. Do this before copying the file into `indata/`'s plain
filenames or handing it off. If any column is missing for the most recent
year(s):

- Tell the user plainly which year(s) and which column(s) are incomplete
  (e.g. "2025 is missing `stir_fr`, `rconpc_usa`, `rconpc_jp`").
- If `stir_au` is one of the missing columns, flag it specifically and point
  the user at the manual Australia step above (re-download from
  <https://www.rba.gov.au/statistics/tables/>, Table F1.1) rather than
  assuming it's just a normal source lag.
- Say what the actual last fully-complete year is (the year before the first
  incomplete one), since that — not `FETCH_END` — is the real "last full year
  of data" the estimation should be told about (see `INSTRUCTIONS.md`).

## 2. Getting the pulled data into the MATLAB model

The MATLAB scripts in `scripts/` read fixed filenames with no year suffix,
e.g. `MainModel1.m` has:

```matlab
[DATA,TEXT] = xlsread('../indata/DataInflShortLongUpdated.xlsx');
```

The master pull writes year-suffixed files (`..._2025.xlsx`) instead, so after a
pull you must do one of the following before re-estimating:

- **Copy/rename** the new file over the plain name the script expects, e.g.:
  ```bash
  cp indata/DataInflShortLongUpdated_2025.xlsx indata/DataInflShortLongUpdated.xlsx
  cp indata/DataInflShortLongConsUpdated_2025.xlsx indata/DataInflShortLongConsUpdated.xlsx
  ```
  (Models 2/3 use the `Cons` workbook; Model 1 uses the non-`Cons` one — see the
  table in `INSTRUCTIONS.md`.)
- **or** edit the `xlsread(...)` line in the `MainModel*.m` script(s) you're
  running to point at the year-suffixed filename directly.

Either way, keep a backup of the previous workbook first — the column layout
(§1 of `INSTRUCTIONS.md`'s customization guide) must stay identical, since the
`MainModel*.m` scripts index columns positionally.

## 3. Changing the ending year

Two independent things determine "how far the data goes":

### a. How far the pull fetches (Python side)

In `scripts/data/master-pull.py`:

```python
FETCH_START = 2021     # first year to (re)fetch
FETCH_END = 2025        # last (complete) year to fetch
```

To extend the pull by a year (e.g. once 2026 data is available), bump
`FETCH_END` to `2026` and re-run the script. Rows for new years are appended to
the workbook automatically (the script extends `year_row` for any fetch year
not already present). Do **not** move `ANCHOR`/`CHAIN_REANCHOR_YEAR` (2020) —
that's the fixed splice point between "trust the existing workbook" and "trust
the freshly fetched source," not something tied to how far forward you fetch.

**`FETCH_END` should be the year prior to the current year, not the current
year itself.** Most sources don't have a complete calendar year of data until
well into the following year, so if you run the pull during 2026, set
`FETCH_END = 2025` (the last *complete* year), not `2026` — fetching a partial
current year will pull in an incomplete annual average for every series. Each
time a new year finishes, bump `FETCH_END` by one to pick it up. See the
"Completeness check" above — the true last complete year can still be earlier
than `FETCH_END` if a specific series lags.

### b. What year the MATLAB estimation stops at

Each `MainModel*.m` script hard-codes the end of the estimation window:

```matlab
T0 = find(Year==1870);
T1 = find(Year==...);
```

After pulling data through a new year and getting it into `indata/` (§2), check
and update `T1` to the new last year in every `MainModel*.m` you plan to run
(the table in `INSTRUCTIONS.md` §"Sample period" lists these; several appendix
variants (`_ReR`, `_1950`, `_df50`, `_var02/05/10/25/50`, `_unrestr`) are
pinned to 2016 on purpose and should generally be left alone). **Read `T1`
directly out of the script rather than assuming a specific year** — this
value moves over time and any year hardcoded in this doc as an example will
go stale. The figure scripts (`MainModel*_MakeFigures.m`) also contain
`find(Year==...)` anchors for plot ranges — update those to match if you want
the new year(s) to show up in the plots.
