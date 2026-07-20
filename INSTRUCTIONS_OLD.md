# Running the Models and Making Plots

Instructions for estimating and plotting **Model 1** (Baseline), **Model 2**
(Convenience Yield), and **Model 3** (Consumption) of the Global r* model.

## Requirements

- MATLAB (developed/tested with R2020a).
- **Always run scripts from inside the `scripts/` directory.** Every script uses
  relative paths (`../indata`, `../results`, `../figures`) and calls
  `addpath Routines`, so it will only resolve correctly when `scripts/` is the
  current MATLAB working directory.

```matlab
cd scripts
```

### Running on the cluster (batch jobs)

To run a script as a batch job on the cluster, use the `matlab20a-batch-withemail`
wrapper (it submits an SGE `qsub` job and emails you when it finishes):

```bash
matlab20a-batch-withemail MEMORY SCRIPT WORKERS
```

- `MEMORY` — gigabytes of RAM for the main program (integer; 1 GB is added
  automatically for MATLAB itself).
- `SCRIPT` — the `.m` file to run (it is launched via `run SCRIPT`).
- `WORKERS` — *optional* per-parallel-worker memory in GB (defaults to `MEMORY`).

Because the wrapper does **not** `cd` into `scripts/` and does not auto-exit
MATLAB, a batch `SCRIPT` should `cd` to the absolute `scripts/` path at the top,
do its work, and call `exit` at the end. Note the output `.mat` files are large
(≈20–36 GB each), so request memory generously (e.g. ≈50–80 GB for the figure
scripts, which load one or two of these files).

## At a glance

| Model | Name             | Run script      | Input data                          | Output `.mat`                  | Figure script                |
|-------|------------------|-----------------|-------------------------------------|--------------------------------|------------------------------|
| 1     | Baseline         | `MainModel1.m`  | `indata/DataInflShortLongUpdated.xlsx`     | `results/OutputModel1.mat`      | `MainModel1_MakeFigures.m`   |
| 2     | Convenience Yield| `MainModel2.m`  | `indata/DataInflShortLongConsUpdated.xlsx` | `results/OutputModel2.mat`      | `MainModel2_MakeFigures.m`   |
| 3     | Consumption      | `MainModel3.m`  | `indata/DataInflShortLongConsUpdated.xlsx` | `results/OutputModel3_new.mat`  | `MainModel3_MakeFigures.m`   |

Each estimation runs `Ndraws = 100000` MCMC draws with `p = 1` VAR lag, so a full
run is slow. These settings are defined at the top of each `MainModel*.m` script
if you want to change them.

### Countries included in each model

All three models estimate and plot the same **18 countries**: US, Germany, UK,
France, Canada, Italy, Japan, Australia, Belgium, Finland, Ireland,
Netherlands, Norway, Switzerland, Sweden, Spain, Portugal, Denmark.

`Nc = numel(codes)` in each `MainModel*.m` (and its matching
`MainModel*_MakeFigures.m`) derives the count from that script's own `codes`
list. If you've edited the country set locally, check `codes` directly in
both the estimation script and its figure script — see "Adding or removing a
country" below; a mismatch between an estimation script and its own figure
script (different country counts/order) is exactly the kind of silent bug
that section warns about.

### Data vintage — which year is actually being used

`T1 = find(Year==...)` in each `MainModel*.m` sets the last year included in
estimation, but the workbook's most recent row is not automatically guaranteed
to be a *complete* year — some series (e.g. World Bank `rconpc_*`, some OECD
rate releases) lag by about a year, so the true "last full year of data" can be
earlier than the workbook's last row. See `DATA.md`'s "Completeness check"
section for how to verify this after a pull.

**Before submitting any batch estimation job, state which year of data is
being used** — i.e. report both `T1` (check its current value directly in the
script rather than assuming a number, since it changes over time) and, if you
just ran a data pull, whether that year was flagged complete or incomplete in
`DATA.md`'s completeness check. Don't submit a multi-hour batch job silently
on a year that was flagged as incomplete without calling that out first.

---

## Step 1 — (Optional) update the data

The estimation reads the Excel data in `indata/`. To re-estimate on new data,
update the relevant workbook before running:

- Model 1: `indata/DataInflShortLongUpdated.xlsx`
- Models 2 & 3: `indata/DataInflShortLongConsUpdated.xlsx` (adds consumption series)

To refresh these workbooks from source (re-pulling the latest years from each
country's data provider) rather than editing them by hand, see
**[DATA.md](scripts/data/DATA.md)** — it covers running the master pull
script, getting its output into `indata/`, and extending the sample to a new
ending year.

## Step 2 — Run a model

Run the model's main script from the `scripts/` directory. Each one clears the
workspace, loads its data, runs the sampler, and saves a single output `.mat`.

```matlab
MainModel1     % -> ../results/OutputModel1.mat
MainModel2     % -> ../results/OutputModel2.mat
MainModel3     % -> ../results/OutputModel3_new.mat
```

### Running estimation as a batch job (from scratch)

Batch drivers `run_model1.m`, `run_model2.m`, `run_model3.m` exist in
`scripts/` and wrap each `MainModel*` script so it can be submitted with
`matlab20a-batch-withemail`. Each driver `cd`s into `scripts/`, runs the
estimation, and (for Models 1 & 2) copies the new `.mat` into `results/18/`
where the figure scripts look; then it `exit`s.

```bash
cd scripts
S=/data/dsge_data_dir/rstarGlobal-18countries/scripts
matlab20a-batch-withemail 130 $S/run_model1.m   # -> results/OutputModel1.mat (+ copy to results/18/)
matlab20a-batch-withemail 110 $S/run_model2.m   # -> results/OutputModel2.mat (+ copy to results/18/)
matlab20a-batch-withemail 130 $S/run_model3.m   # -> results/OutputModel3_new.mat
```

These are **slow** (100 000 MCMC draws) and **memory-hungry** (the saved `.mat` is
20–36 GB and is held in RAM during sampling), hence the large memory requests.
Figure 2 of Model 1 additionally needs `results/OutputModel1_var01.mat`, produced
by submitting `MainModel1_var01.m` the same way. Monitor jobs with `qstat` and
check exit status / peak memory afterward with `qacct -j <jobid>`.

To estimate all main-body specifications in one pass (Models 1, 1_var01, 2, 3),
run `estimateAll` instead. Set `estimateAppendices = 1` inside that script to
also estimate the appendix specifications.

> **Important — output location for figures.** The figure scripts for Models 1
> and 2 load their input from `../results/18/`, but `MainModel1.m` / `MainModel2.m`
> save to `../results/`. After re-estimating, copy the new output into
> `results/18/` (overwriting the old file) so the figure scripts pick it up:
>
> ```matlab
> copyfile('../results/OutputModel1.mat', '../results/18/OutputModel1.mat')
> copyfile('../results/OutputModel2.mat', '../results/18/OutputModel2.mat')
> ```
>
> Model 3 is different: its figure script loads `../results/OutputModel3_new.mat`
> directly, so no copy is needed (just make sure the file is in `results/`). The
> repository already ships pre-estimated outputs in `results/18/`, so you can
> skip Step 2 entirely and go straight to plotting if you only want the figures.

## Step 3 — Make the plots

The figure scripts are organized into `%%` cell blocks. **Always run the
preliminaries block first** — it loads the `.mat` output and computes the
quantiles every later block relies on — then run whichever figure block(s) you
want.

In the MATLAB editor, click a block and press **Ctrl+Enter** (Cmd+Enter on Mac)
to run just that block, or run the whole script with `run`.

Figures are written as PDFs to `../figures/` (main body) and
`../figures/appendix/` (appendix) via the `printpdf` helper. A few Model 1
figures are also saved as `.eps` into the `scripts/` folder.

### Model 1 — `MainModel1_MakeFigures.m`

1. Run **`Plot preliminaries: load, sort, and get quantiles`** first.
2. Then run any figure block:
   - `Figure 1` — Trends in Global and U.S. Real Rates (also saves `figure1.eps`)
   - `Figure 2` — Trends under alternative priors (loads `../results/OutputModel1_var01.mat`)
   - `Figure 3a` / `Figure 3b` — Short-term real rate trends vs. observables (3b saves `figure3.eps`)
   - `Figure 4a` / `Figure 4b` — Inflation trends vs. observables
   - `Figure 5` — MY regression fit
   - Appendix: `A1` (term spreads), `A2` (country-specific trends)

### Model 2 — `MainModel2_MakeFigures.m`

1. Run **`Preliminaries`**, then the extract/sort/quantiles blocks above the figures.
2. Then run any figure block:
   - `Figure 6a` — r-bar^w
   - `Figure 6b` — r-bar^w and convenience yield
   - `Figure 6c` — r-bar^w and m-bar^w
   - Appendix: `A13`–`A17`
- Term-spread appendix figures live in the companion script `MainModel2_MakeFigures_Spreads.m`.

### Model 3 — `MainModel3_MakeFigures.m`

1. Run **`Preliminaries`**, then the extract/sort/quantiles blocks above the figures.
2. Then run any figure block:
   - `Figure 7a` — Rbar and convenience yield
   - `Figure 7b` — Rbar and g-bar
   - `Figure 7c` — Rbar and beta-bar
   - Appendix: `A18` (Global and U.S. real rate trends)

### Make all figures at once

`makeFigures.m` runs every `*_MakeFigures` script in sequence (Models 1, 2, 3 plus
appendix specifications). Use it once the corresponding outputs exist in `results/`
(and `results/18/` for Models 1 and 2).

### Making figures as batch jobs

In some copies of this pipeline, batch drivers `make_figs_m1.m` /
`make_figs_m2.m` / `make_figs_m3.m` wrap the three main-body figure scripts so
they run non-interactively (each `cd`s into `scripts/`, runs the whole
`*_MakeFigures.m` top-to-bottom, and `exit`s). **These do not exist in this
checkout** — only the `run_model1/2/3.m` estimation drivers do (see Step 2).
If you want the same non-interactive figure-generation capability here,
create `make_figs_m1/2/3.m` following that pattern (`cd` to the absolute
`scripts/` path, `run('MainModel*_MakeFigures.m')` wrapped in `try/catch`,
`exit`), then submit them the same way as `run_model*.m`:

```bash
cd scripts
S=/data/dsge_data_dir/rstarGlobal-18countries/scripts
matlab20a-batch-withemail 80 $S/make_figs_m1.m   # loads OutputModel1.mat (+ var01)
matlab20a-batch-withemail 48 $S/make_figs_m2.m   # loads OutputModel2.mat
matlab20a-batch-withemail 110 $S/make_figs_m3.m  # loads OutputModel3_new.mat (36 GB -> needs ~110 GB)
```

The memory figures above are sized to the `.mat` each script loads (the file is
expanded in RAM, so request well above the on-disk size — Model 3's 36 GB load needs
~110 GB). PDFs land in `figures/` and `figures/appendix/`.

## Step 4 — (Optional) tables

Two table scripts write LaTeX tables to `tables/`. Both must be run from
`scripts/` and load the 20–30 GB outputs from `results/18/`, so on the cluster
submit them as batch jobs rather than running interactively.

### `Tables.m` — Global and US r\* summary tables (blog replication)

Builds `tables/GlobalUS_Model1.tex`, `tables/GlobalUS_Model2.tex`, and
`tables/GlobalUS_Combined.tex` (both models stacked as panels in one table):
the change in global and US r-bar over 1990–2019 and 2019–2024, and (Model 2)
its decomposition into m-bar ("other") and the convenience-yield
contribution.
This is the layout of the Liberty Street Economics table **"Pre- and
Post-COVID Changes in R\*"**
(<https://libertystreeteconomics.newyorkfed.org/2026/02/the-post-pandemic-global-r/>),
and was verified (2026-07-09) to reproduce that table exactly from the
Sep 2025 estimates. Note: the `results/18/Output*.mat` in **this** checkout
are a July 2025 re-estimation (different MCMC draws), so expect the output to
differ from the blog's numbers by small amounts — that is sampling noise, not
a bug. Each cell reports the posterior median, the **90%**
equal-tailed interval (5th/95th percentiles — what the blog displays), and
`{P(change < 0)}`, from which the blog's significance stars follow:
\*/\*\*/\*\*\* if the posterior probability that the change is below zero
(1990–2019) or above zero (2019–2024) exceeds 0.90/0.95/0.975.

Conventions worth knowing (both were bugs once):

- **Set `Quant` only after `load()`.** The `Output*.mat` files contain their
  own saved 5-element `Quant` variable from the estimation script, which
  silently overwrites anything set before the load. `Tables.m` redefines
  `Quant` after each load for exactly this reason — keep it that way.
- **US decomposition sign convention.** There is no US-specific m-bar, so the
  US "other" row reuses the global m-bar, and the US convenience-yield row is
  defined residually via `Cy_bar_us = Cy_bar - (Rshort_bar_us - Rshort_bar)`,
  so that `Rshort_bar_us = M_bar - Cy_bar_us` and the displayed rows sum
  per-draw to the US r\* change.

**Choosing the comparison years (edit before running).** The periods over
which the r\* changes are computed are hard-coded in `Tables.m` and must be
set to what you want — they are **not** parameters passed in from anywhere:

- The year anchors: the `t_start1/t_end1` (first column) and `t_start2/t_end2`
  (second column) lines, set via `find(Year == <year>)`. These appear
  **twice** — once in the Model 1 section and once in the Model 2 section —
  and both must be edited together. The years must lie inside the estimation
  sample (as shipped, 1870–2024): `find` returns empty for a year outside the
  sample and the script will error.
- The column headers: the `header = {'', '1990-2019', '2019-2024'}` lines
  (also once per section) are plain text and do not update automatically —
  change them to match the years you set.
- Optionally the credible band: `Quant = [0.050 0.500 0.950]` (once per
  section, after each `load` — see the warning above) if you want something
  other than the blog's 90% interval.

To run as a batch job, wrap it in a small driver script (the batch wrapper
does not `cd` into `scripts/` or auto-exit MATLAB — same pattern as the
`run_model*.m` drivers): a `.m` file that `cd`s to the absolute `scripts/`
path, calls `Tables`, then `exit`s. Submit it with
`matlab20a-batch-withemail 80 <driver>.m` — it loads Models 1 and 2
sequentially and takes ~5–6 minutes with a peak of ~44 GB.

### `makeTables.m` — appendix tables (A1, A2)

Builds the multi-band appendix tables. Run the `Table A1a ...` block for the
baseline (Model 1) decomposition and the `Table A1b ...` block for the
convenience-yield (Model 2) decomposition.

> **Known bug:** the US −cy row of `makeTables.m`'s Model 2 table (A1b) has a
> flipped sign (it prints +0.85 where the blog table has −0.85), from the old
> `Cy_bar_us = (Rshort_bar_us − Rshort_bar) − Cy_bar` definition. `Tables.m`
> uses the corrected definition; prefer it for the r\* change/decomposition
> numbers.

---

# Customization guide

This section documents every change you are likely to need: swapping the data,
adding/removing a country, changing the sample period, the sampler settings, and
the priors. **Read the whole "adding a country" checklist before doing it — the
country set is wired into several blocks, and missing one causes a silent
mis-indexing or a crash (this is exactly how Denmark got left half-wired in
other copies of this pipeline — in this checkout, all three models are
verified consistent at 18 countries).**

## 1. The data files and their column layout

Each model reads one Excel workbook from `indata/` at the top of its `MainModel*.m`
(the `xlsread(...)` line). `Year` is column 1; `X = DATA(:,2:end)` drops it, so
**`X` column = spreadsheet column − 1**.

**Baseline file — `DataInflShortLongUpdated.xlsx` (Models 1 and var01).**
This layout is **irregular** — do not assume a fixed stride:

| Spreadsheet cols | `X` cols | Contents |
|---|---|---|
| 1 | — | `year` |
| 2–22 | 1–21 | `cpi/stir/ltir` for us, de, uk, fr, ca, it, jp — **3 cols each, no consumption** |
| 23 | 22 | `baa_usa` (a one-off US BAA bond-yield column — skipped by the models) |
| 24–63 | 23–62 | `cpi/stir/ltir/rconpc` for au, be, fi, ie, nl, no, ch, se, es, pt — **4 cols each** |
| 64–67 | 63–66 | `cpi/stir/ltir/rconpc` for **dk** (Denmark) |

Because of the `baa_usa` column and the 3-vs-4 column split, **Model 1 must use
explicit `X(:,col)` indexing** (as in `MainModel1_var01.m`), not a fixed
`country_start` stride. The consumption (`rconpc_*`) columns exist in this file but
Model 1 does not use them.

**Consumption file — `DataInflShortLongConsUpdated.xlsx` (Models 2 and 3).**
This one **is** uniform: `cpi/stir/ltir/rconpc`, 4 columns per country, in country
order. That is why Models 2/3 can use the `country_start = [0,4,8,…]` loop.

### To update the numbers / add a new time period
Edit the relevant workbook in `indata/` in place (keep the header row and column
order), then re-estimate. To extend the sample forward, add rows (years) at the
bottom and update the sample-end year in §3. To do this via the automated pull
instead of by hand, see **[DATA.md](scripts/data/DATA.md)**.

## 2. Adding or removing a country

`Nc = numel(codes)` is derived, but the country set appears in several blocks that
must all agree. Using **Model 1** (`scripts/MainModel1.m`) as the reference, to add
a country you must touch **all** of these:

1. **Data must contain it.** Confirm the workbook has `cpi/stir/ltir` columns for
   the country (see §1 for the baseline layout). Note its `X` column numbers.
2. **`codes` list** (near line 79) — add the 2-letter code.
3. **Data loading** — add explicit assignments with the correct columns, e.g. for
   Denmark: `Price_dk = X(:,63); Stir_dk = X(:,64); Ltir_dk = X(:,65);`
   (`Infl_*` is then computed for every code by the inflation loop).
4. **`Y = [ … ]` observation matrix** — add the country to **all three** blocks:
   `Stir_xx…`, `Infl_xx…`, `Ltir_xx…`.
5. **`Mnem = { … }` labels** — add `'Stir_xx'…`, `'Infl_xx'…`, `'Ltir_xx'…` in the
   same three blocks (order must match `Y`).
6. **`Ctr = [ … ]` observation-loading matrix** — add one row per series. Each row
   is **three 0/1 flags** = whether that series loads on the three **world/common**
   trends, in the order `[world real-short-rate, world inflation, world term-spread]`
   (the text after `%` is only a comment). Copy the pattern used by the other
   countries of the same type: **Stir → `1 1 0`, Infl → `0 1 0`, Ltir → `1 1 1`**.

   These three flags form the first three columns of the observation matrix
   `C = [Ctr Cadd1 Cadd2 Cadd3 Ccyc]`, which the Kalman filter
   `KF(y, C, R, A, Q, S0, P0)` uses as `y = C·state`. The state order is documented
   at the `SC0tr`/`S0tr` comment (`rs_wrd pi_wrd ts_wrd  rs_idio pi_idio ts_idio`),
   and `CommonTrends = States(:,1:r,:)` — so `CommonTrends(:,1:3,:)` are exactly those
   three world trends the figures plot. The loadings are the standard identities:
   short nominal rate = real + inflation, inflation = inflation, long nominal rate =
   real + inflation + term spread. `Cadd1/Cadd2/Cadd3` are the **idiosyncratic**
   (per-country) counterparts and mirror the same per-series pattern
   (rs → Stir & Ltir, pi → all three, ts → Ltir only), so a new country row in `Ctr`
   plus its `Cadd*` columns just repeats that structure.
7. **Check the hard-coded trend-loading matrices `Cadd2` / `Cadd3`** (search for
   `zeros(n,18)` / `eye(18)` / `Cadd2(19:36,…)` / `Cadd3(37:54,…)`). `Cadd1` is
   written in terms of `Nc` and updates automatically, but **`Cadd2` and `Cadd3`
   hard-code `18` and the `1:18 / 19:36 / 37:54` row ranges (= `Nc / 2·Nc / 3·Nc`).**
   For a different country count, replace `18`→`Nc`, `19:36`→`Nc+1:2*Nc`,
   `37:54`→`2*Nc+1:3*Nc` (mirroring `Cadd1`), or edit the literals to the new count.
8. **Figure scripts** — the matching `MainModel*_MakeFigures.m` derives
   `Nc = numel(codes)` from its **own** `codes` list, so update that list to the same
   country set, and check for any hard-coded per-country arrays (e.g. the
   `country_colors` matrix in the country-panel blocks of `MainModel1_MakeFigures.m`
   — note that file redefines `codes`/`country_colors` locally in more than one
   `%%` block, so check *every* occurrence, not just the first).

**Removing a country** is the same list in reverse — delete its entries from
`codes`, the data loads, `Y`, `Mnem`, `Ctr`, fix `Cadd2/Cadd3`, and the figure
`codes` list.

**Models 2 and 3** follow the same checklist, except the data load is the
`country_start = [0,4,8,…]` loop (uniform consumption file): to add a country,
extend `codes` **and** append the next start column to `country_start`, then update
`Y` / `Mnem` / `Ctr` (and the `Cadd*` dimensions) as above. In this checkout, Models
1, 2, and 3 all estimate the same 18 countries (see "Countries included in each
model" above) — if you change the country set in one model, the other two, and
their figure scripts, need the same change to stay consistent.

## 3. Sample period

In each `MainModel*.m`:
- Estimation window: `T0 = find(Year==1870);` and `T1 = find(Year==...);` —
  check the current value of `T1` directly in the script rather than assuming
  a specific year (see "Data vintage" above).
- Presample used for the moment print-out: `T0pre = find(Year==1870);`,
  `T1pre = find(Year==1899);`

The figure scripts also contain `find(Year==…)` anchors (e.g. plot start/end years);
update those to match if you change the window.

## 4. Sampler settings

At the top of each `MainModel*.m`:
- `Ndraws = 100000;` — number of MCMC draws (the main cost; lower it for quick tests).
- `p = 1;` — number of VAR lags in the cyclical component.

## 5. Output file name

`filename = '../results/OutputModel1.mat';` near the top of each `MainModel*.m`
controls where the estimate is saved.

## 6. Isolated re-runs

Some copies of this pipeline have a `claude_test/` sandbox — absolute-path
copies of the `MainModel*`/`*_MakeFigures` scripts that read/write only under
`claude_test/`, plus a `run_all.sh` that submits the full estimation+figures
pipeline unattended. **This checkout does not have a `claude_test/`
directory** — re-running `MainModel*.m`/`run_model*.m` here writes directly to
`results/` (and `results/18/`, per Step 2's copy step), there is no isolated
sandbox that leaves `results/` untouched. If you want that isolation, create a
`claude_test/`-style copy following the pattern above before re-running
estimations you don't want to risk overwriting the shipped results.
