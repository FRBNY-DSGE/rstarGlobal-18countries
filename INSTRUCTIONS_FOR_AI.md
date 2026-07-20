# Running the Global r\* Model (18 countries) — Updated

Estimate and plot **Model 1** (Baseline), **Model 2** (Convenience Yield), and
**Model 3** (Consumption) — plus **`var01`** (Model 1 under a disperse trend
prior, needed for Figure 2). All models are **18 countries** (the 17 in the
published paper **plus Denmark**).

> This supersedes the original `INSTRUCTIONS.md` (renamed **`INSTRUCTIONS_OLD.md`**)
> for anyone re-running the model.
> It reflects the fixes made during the 2025-vintage run; see **[Fixes.md](Fixes.md)**
> for the full list of what was broken as-shipped and how it was repaired.

> **State of the repo (as of the 2025-vintage run).** The data currently wired in
> runs **1870–2025**, all four specs have been estimated, and the full figure set
> (20 main + 90 appendix PDFs) has been produced. **Read Fixes.md before touching
> the observation matrices or data files** — several things (the data layout,
> Denmark wiring, the var-family prior knob) are subtle.

---

## 0. If you are here to add a new year of data (e.g. 2025 → 2026)

This is the common case. The scripts hold *no* hard-coded end-year anchors except
the two knobs below, so updating the sample is short. **Full checklist:**

1. **Pull the new year.** In `scripts/data/master-pull.py` set `FETCH_END = 2026`
   (keep `ANCHOR`/2020 fixed — that is the fixed splice point, not the end year),
   then run it. See **[scripts/data/DATA.md](scripts/data/DATA.md)** for details,
   dependencies, and caveats (OECD caching, the UK long-rate `playwright` step).
   > `FETCH_END` must be the last *complete* calendar year. If you run this during
   > 2026, use `2025`; only bump to `2026` once 2026 is a finished year with
   > published annual data. Otherwise you pull a partial-year average.

   > **Australia short rate — auto-fetched now, with a hand-download fallback.**
   > Every series (Australia included) is fetched from source automatically:
   > `sr_australia()` pulls the RBA **F1.1** money-market CSV directly from
   > <https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv> and takes the
   > annual mean. **You normally do nothing.** Only if that fetch fails does the
   > pull log `[warn ] RBA F1.1 direct fetch failed …` and fall back to the local
   > file `indata/raw/f1.1-data.csv`. If the fetch failed **and** that local file
   > is stale/missing, the AI should **stop and ask the user** to refresh it:
   > - **What:** RBA statistical table **F1.1 – Interest Rates and Yields – Money
   >   Market** (the CSV, monthly).
   > - **Where from:** <https://www.rba.gov.au/statistics/tables/> → scroll to
   >   **"Interest Rates and Yields – Money Market – Monthly – F1.1"** → click the
   >   **"Data [CSV]"** button.
   > - **Where to put it:** overwrite `indata/raw/f1.1-data.csv`, exactly as
   >   downloaded — an 11-row header (`skiprows=11`) then monthly
   >   `DD/MM/YYYY,value,…` rows (the loader takes the date + first rate column,
   >   Cash Rate Target, → annual mean). Don't reformat or re-order columns.
   >
   > See `scripts/data/DATA_UPDATED.md` §1b for detail.

   Output lands in the pull's aggregate folder as
   `DataInflShortLongConsUpdated_2026.xlsx` (and a `rconpc`-stripped sibling).

2. **Wire it to the plain filename** the MATLAB scripts read. Back up the current
   plain file first, then copy the new vintage over it:
   ```bash
   cd indata
   cp DataInflShortLongConsUpdated.xlsx DataInflShortLongConsUpdated_pre2026backup.xlsx
   cp <pull-output>/DataInflShortLongConsUpdated_2026.xlsx DataInflShortLongConsUpdated.xlsx
   ```
   > **All three models read the same 74-column Cons workbook** —
   > `DataInflShortLongConsUpdated.xlsx`. Model 1 reads it too (see §2); you do
   > **not** need the non-cons file. Keep the column layout identical — the scripts
   > index columns *positionally*.

3. **Bump the sample-end year** in every `MainModel*.m` you plan to run:
   ```matlab
   T1 = find(Year==2025);   % -> change 2025 to 2026 in MainModel1.m,
                            %    MainModel2.m, MainModel3.m, MainModel1_var01.m
   ```
   (Leave the appendix `_var02/05/…`, `_ReR`, `_1950`, `_df50`, `_unrestr` specs
   alone — those are pinned to 2016 on purpose.)

4. **`Data_MY.xlsx` needs no action.** The demographic "MY" regressor used only in
   Model 1's Figure 5 ends in **2016**; the figure script already regresses on the
   overlapping years and lets the fitted line stop at 2016 (see §5). Nothing to
   update unless you actually obtain newer MY data.

5. **Estimate, then plot** — §3 and §4 below. The public `update/` figures + CSVs
   regenerate automatically inside `make_figs_m1` (§4); no separate step.

6. **If you make the tables, bump `Tables.m`'s year anchors.** `Tables.m` (§6)
   hard-codes the summary window as `find(Year==2019)`→`find(Year==2025)`; update
   the `2025` end-year anchors (one block per model — Models 1, 2, 3) to the new
   year. `makeTables.m` and the figure scripts have no other end-year anchors.

That's it. There are no other year references to chase in the figure scripts.

---

## 1. Requirements & the cluster workflow

- MATLAB (developed/tested with R2020a).
- **Always run scripts from inside `scripts/`.** Every script uses relative paths
  (`../indata`, `../results`, `../figures`) and calls `addpath Routines`.
- The estimations are **slow** (`Ndraws = 100000` MCMC draws, `p = 1` VAR lag;
  Model 3 ≈ 0.30 s/draw ≈ 8–9 h, Models 1/2/var01 ≈ 0.14 s/draw ≈ 4 h) and the
  saved `.mat` files are **large** (20–38 GB, held in RAM during sampling).

Submit each job to the cluster with the `matlab20a-batch-withemail` wrapper
(`MEMORY` in GB; it emails you on completion):
```bash
matlab20a-batch-withemail MEMORY SCRIPT.m
```
Ready-made batch drivers live in `scripts/` (each `cd`s to `scripts/`, does its
work, and `exit`s):

| Driver | Runs | Output | Copies to `results/18/`? |
|---|---|---|---|
| `run_model1.m`        | `MainModel1`        | `results/OutputModel1.mat`       | yes |
| `run_model1_var01.m`  | `MainModel1_var01`  | `results/OutputModel1_var01.mat` | no (fig loads from `results/`) |
| `run_model2.m`        | `MainModel2`        | `results/OutputModel2.mat`       | yes |
| `run_model3.m`        | `MainModel3` (version B, default)   | `results/OutputModel3_new.mat`   | no |
| `run_model3_A.m`      | `MainModel3_A` (alt inflation prior)| `results/OutputModel3_A.mat`     | no |
| `make_figs_m1/2/3.m`  | `MainModel*_MakeFigures` | PDFs in `figures/` (+ `appendix/`); **`make_figs_m1` also writes `update/` PNGs + CSVs** | — |
| `run_tables.m`        | `Tables.m`          | `tables/GlobalUS_*.tex` (incl. slide-format) | — |

Monitor with `qstat`; check peak memory / exit status afterward with
`qacct -j <jobid>`.

---

## 2. Data files & column layout

`Year` is column 1; `X = DATA(:,2:end)` drops it, so **`X` col = spreadsheet col − 1**.

**The models all read `DataInflShortLongConsUpdated.xlsx` — 74 columns, uniform:**
`cpi / stir / ltir / rconpc`, 4 columns per country, in country order, with a
single one-off `baa_usa` column after Japan (spreadsheet col 30). Denmark occupies
the last four columns (71–74). The loader is a loop:
```matlab
country_start = [0, 4, 8, 12, 16, 20, 24, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69];
```
Note the stride jumps 24→29 (that skip is the `baa_usa` column). Model 1 uses this
same loop and simply ignores the `rconpc_*` columns; Model 2 uses `Baa_us` as an
extra observable and appends it to `Y`.

> There is also a 56-column *non-cons* file (`DataInflShortLongUpdated_*.xlsx`,
> `rconpc` stripped). **Do not feed it to Model 1** — the loop above expects the
> 74-column layout and would mis-index against 56 columns.

## 3. Estimate

```bash
cd scripts
S=/data/dsge_data_dir/rstarGlobal-18countries/scripts
matlab20a-batch-withemail 130 $S/run_model1.m
matlab20a-batch-withemail 130 $S/run_model1_var01.m
matlab20a-batch-withemail 110 $S/run_model2.m
matlab20a-batch-withemail 130 $S/run_model3.m
```
To estimate every main-body spec in one MATLAB session instead, run `estimateAll`
(set `estimateAppendices = 1` inside it to also do the appendix specs).

**Sanity check a fresh submission:** an out-of-bounds or parse error surfaces in
the first ~6 s (before the MCMC loop). `qacct -j <jobid>` showing a 6-second run
means it crashed at setup — read `run_model*.e<jobid>`. A healthy job prints
`Nth draw of 100000 …` lines. A cheap pre-submit check:
`matlab -nodisplay -r "checkcode('MainModel1.m','-string'), exit"`.

## 4. Make the figures

The figure scripts load the estimates and write PDFs to `figures/` (main body)
and `figures/appendix/`. Models 1 & 2 load from **`results/18/`** (the run drivers
copy the `.mat` there); Model 3 loads `results/OutputModel3_new.mat` directly.

Run them **after** the corresponding estimates exist. To make the whole pipeline
unattended, submit the figure jobs *held* on the estimation jobs so each releases
automatically when its estimate finishes (SGE `-hold_jid`):
```bash
S=/data/dsge_data_dir/rstarGlobal-18countries/scripts
ML="/apps/matlab20a/bin/matlab -singleCompThread -nodesktop -nodisplay -nosplash -r"
qsub -m e -cwd -N make_figs_m2 -hold_jid <m2_jobid>              -l h_vmem=65G  -b y $ML "run $S/make_figs_m2.m"
qsub -m e -cwd -N make_figs_m1 -hold_jid <m1_jobid>,<var01_jobid> -l h_vmem=81G -b y $ML "run $S/make_figs_m1.m"
qsub -m e -cwd -N make_figs_m3 -hold_jid <m3_jobid>              -l h_vmem=111G -b y $ML "run $S/make_figs_m3.m"
```
> **Model 1's figure job must wait on BOTH the baseline and `var01`**, because
> Figure 2 overlays the two priors. If `var01` is not available, Figure 2 is
> skipped (guarded) but the rest of the M1 figures still generate.

Each `make_figs_*` prints `FIGURES_DONE model N` on success; if a block errors it
logs `FIGURE SCRIPT ERROR` (in `make_figs_m*.e<jobid>`) and still exits — check
for that marker to confirm a run was complete, not just finished.

Figure memory ≈ the `.mat` size expanded in RAM: M1 ~80 GB, M2 ~64 GB, M3 ~110 GB.

> **`make_figs_m1` also writes the public `update/` artifacts.** The `fig1` and
> `fig3b` blocks of `MainModel1_MakeFigures.m` additionally `saveas` the repo
> PNGs (`qRshort_bar_us_global_m1.png`, `qRshort_bar_m1.png`) and write the three
> `update/*.csv`, from the *same* figure handles/quantiles that make the paper
> PDFs — one source for the paper figure and its GitHub copy. This replaced the
> old standalone `make_update.m`/`run_update.m`; do not reintroduce a second
> script for these figures (that duplication once let `fig1` clip at 2024 while
> the PNG ran to 2025).

## 5. The `Data_MY.xlsx` / Figure 5 caveat

`Data_MY.xlsx` (Model 1 Figure 5's demographic regressor, G7 only) ends in **2016**.
The estimation sample is longer, so `MainModel1_MakeFigures.m` runs the MY
regression on the overlap and NaN-pads the fitted line past 2016 (helpers `nMY`,
`padMY`, `fitMY`). The fitted MY line therefore stops at 2016 by design — this is
expected, not a bug. Every other figure uses the full `Year` vector and extends
automatically.

## 6. Optional: tables

Two table scripts, both writing LaTeX into `tables/`:

- **`makeTables.m`** — the appendix decomposition tables (block `Table A1a` for the
  Model-1 decomposition, `Table A1b` for Model 2).
- **`Tables.m`** — Global vs US r\* summary tables: the change in r-bar over
  1990–2019 and 2019–2025 (posterior median, 90% interval, P(change<0) per cell),
  with each model decomposed — Model 2 into r\*/−cy/"other" (m-bar), Model 3 into
  r\*/g/β/−cy. It writes:
  - `tables/GlobalUS_Model1.tex`, `GlobalUS_Model2.tex`, `GlobalUS_Model3.tex`
  - `tables/GlobalUS_Combined.tex` — all three model panels in one table
  - `tables/GlobalUS_Levels.tex` — end-of-sample **level** of global/US r-bar
    (Fig-1 value) for Models 1 & 2
  - `tables/GlobalUS_Model3_A.tex` — the **alternative** Model 3 (version A, the
    undocumented `2` inflation prior); the main `GlobalUS_Model3.tex` is version B
  - **slide-format (star) tables** — Layout A (same rows/cols, cells
    `median^{stars} (lo,hi)`) appended into each `GlobalUS_Model{1,2,3}.tex`, plus
    Layout B (deck replica) in `GlobalUS_SlideReplica.tex`, at 90% & 95% CI. Stars
    mark P(change in the expected direction) > .90/.95/.975; the consuming document
    needs `\usepackage{makecell}`.

  It `load`s the full `results/18/OutputModel{1,2}.mat` **and**
  `results/OutputModel3_new.mat` (37 GB), so run it as a batch job with ~110 GB
  via its driver:
  ```bash
  S=/data/dsge_data_dir/rstarGlobal-18countries/scripts
  matlab20a-batch-withemail 110 $S/run_tables.m
  ```
  > It hard-codes the end year as `find(Year==2025)` (one block per model) — bump
  > those when you move to a new sample year (see §0 step 6). `Tables.m` is a
  > companion to `makeTables.m`, not a replacement.

---

## 7. Customization

- **Sampler:** `Ndraws` and `p` at the top of each `MainModel*.m` (lower `Ndraws`
  for quick tests).
- **The var-family prior.** `MainModel1_var01`…`_var50` are the baseline with one
  knob: the `SC0tr` divisor (the prior variance of trend innovations).
  `var01 = /1` (most disperse), `var02 = /2`, `var05 = /5`, `var10 = /10`,
  `var25 = /25`, `var50 = /50`; baseline `MainModel1 = /100` (tightest). To make a
  new variant, copy `MainModel1.m`, change the header, `filename`, and that one
  divisor — nothing else. (Do **not** copy the old `MainModel1_var02…50.m` data
  loaders: they still use the obsolete 67-column layout; regenerate from
  `MainModel1.m` instead. See Fixes.md §5.)
- **The Model 3 inflation-trend prior (A vs B).** `MainModel3.m` (default, version
  **B**) sets the `SC0tr` inflation entry to `sqrt(2)` → prior 1/50, matching
  Models 1 & 2 and the paper. `MainModel3_A.m` uses `2` → 1/25 (the original,
  undocumented setting) and is the alternative, with its own `figures/model3_A/`
  and `GlobalUS_Model3_A.tex`.
- **Figure end-years must be `Year(end)`, never a literal.** A hard-coded
  `axis([1880 2024 …])` in one block once clipped `fig1` a year behind every other
  figure. If a single figure ends a year short of the rest, look for a literal
  year in its `axis(...)`/`xlim(...)`.
- **Adding / removing a country.** The country set is wired into several blocks that
  must all agree — get one wrong and you get a silent mis-index or a crash (this is
  exactly how Denmark ended up half-wired). For each model you must touch **all** of:
  1. `codes` (drives `Nc`) and the `country_start` loop (it already has 18 entries;
     add the next start column for a new country).
  2. `Y`, `Mnem`, and `Ctr` — add the country's `Stir/Infl/Ltir` rows to **all
     three** blocks, in the same order. Loadings: `Stir → 1 1 0`, `Infl → 0 1 0`,
     `Ltir → 1 1 1` (Model 2 adds a 4th convenience-yield flag: `-1/0/-1`, and its
     `Baa_us` observation stays last).
  3. **Model 1 only:** `Cadd2`/`Cadd3` hard-code the count `18` and the row ranges
     `1:18 / 19:36 / 37:54`. Update those to the new `Nc` (Model 2/3 write these in
     terms of `Nc` and auto-scale).
  4. The matching `*_MakeFigures.m`: its own `codes` list **must equal the model's**
     (it derives `Nc` and uses it as a stride to slice per-country trends — a
     mismatch silently corrupts them), and any hard-coded `country_colors` array
     must have at least `Nc` rows.
