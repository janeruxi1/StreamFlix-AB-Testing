# Data

Two public, real randomized experiments, plus one simulator used only where the true answer
has to be known.

## 1. Hillstrom e-mail test (Part 1, notebooks 01–03)

Kevin Hillstrom's MineThatData E-Mail Analytics and Data Mining Challenge (March 2008):
64,000 customers randomized in equal thirds to a men's merchandise e-mail, a women's
merchandise e-mail, or no e-mail, with visits, purchases and spend over the following two
weeks.

**Not committed.** The file is free to download, but its terms of use are not stated, so it is
not redistributed here. To reproduce Part 1:

1. Download the CSV linked from the challenge post on the MineThatData blog (March 2008):
   <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv>.
2. Save it unchanged in `data/raw/` as
   `Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
   (a `.csv.gz` of the same file also works).
3. Check it is the same file the results were computed on:

   | | |
   |---|---|
   | Size | 3,964,977 bytes |
   | SHA-256 | `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` |

Without the file, notebooks 01–03 run on a synthetic stand-in with the same columns and
planted effects (`src/data/hillstrom.py: make_synthetic`). That is what CI does
(`HILLSTROM_SYNTHETIC=1`). Stand-in runs write their results to a scratch folder, never to
`reports/metrics/`.

## 2. Upworthy Research Archive (Part 2, notebooks 04–05)

J. Nathan Matias, Kevin Munger, Marianne Aubin Le Quere and Charles Ebersole, "The Upworthy
Research Archive, a time series of 32,487 experiments in U.S. media", *Scientific Data* 8, 195
(2021). Licensed **CC BY 4.0**. Every headline test Upworthy ran between January 2013 and
April 2015: per package (headline plus image), its impressions and clicks.

**Committed, in compact form,** in `data/upworthy/`: one Parquet file per sample
(`exploratory`, `confirmatory`, `holdout`), keeping only the columns the analysis uses
(`clickability_test_id`, `created_at`, `headline`, `eyecatcher_id`, `impressions`, `clicks`,
`winner`). Changes from the original: columns dropped and `created_at` parsed to a timestamp;
no rows removed or values edited.

To rebuild them from the originals (the archive's OSF project, <https://osf.io/jd64p/>, files
dated 03.12.2020), put the three `upworthy-archive-{sample}-packages-03.12.2020.csv` files in `data/raw/` and run:

```bash
python -c "from src.data import upworthy; [upworthy.build_compact(s) for s in ('exploratory', 'confirmatory', 'holdout')]"
```

In June 2024 the maintainers reported possible broken randomization for tests run from
25 June 2013 to 10 January 2014. The 2020 files do not flag those tests, so the window is
applied by each test's start date (`src/data/upworthy.py`). Notebook 04 checks whether a
sample-ratio check finds the problem without being told the dates.

## 3. Simulator (the StreamFlix case study and the unit tests)

`src/data/simulate.py` generates a homepage test with a planted assignment bug and a known
effect; `make_synthetic` in each data module generates tests with known effects. Used only to
check that the methods recover a known truth, never for a reported business result. Its schema
and planted ground truth are in [`streamflix.md`](./streamflix.md).
