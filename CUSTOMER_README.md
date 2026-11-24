# Profiling your tables for a synthetic benchmark dataset

This package profiles the tables your SQL workload uses and records their statistical shape: sizes, distributions,
skew, and how the tables join. From that we generate a synthetic copy at a larger scale (3× by default) and run the
benchmark on it. Your data never leaves your workspace. You run one notebook, review the two files it writes, and
send those files back.

**The notebook only reads your tables. It doesn't create, change or delete any table.** Its only outputs are
files in a workspace folder of your choice.

---

## 1. What's in the package

| Path | What it is |
|---|---|
| `01_profile_and_derive.py` | The notebook you run |
| `synth/` | The Python code the notebook imports (keep it next to the notebook) |
| `wheels/dbldatagen-0.4.0.post2-py3-none-any.whl` | The data-generation library, installed into the notebook session only |
| `CUSTOMER_README.md` | This guide |

**The SQL workload isn't included; you already have it.** The notebook needs it to know which tables, joins and
filters matter. Before running, put the `.sql` files in a folder (one or more statements per file; subfolders are
read too). Either:

- create a `queries/` folder next to the notebook and upload the files there (the default), or
- point the `queries_path` widget at a folder or a single `.sql` file elsewhere in the workspace.

## 2. Requirements

- **Compute:** serverless notebook compute (recommended), or a Unity Catalog cluster on Databricks Runtime 15.4 LTS
  or later.
- **Permissions:**
  - `USE CATALOG` / `USE SCHEMA` and `SELECT` on the source tables;
  - write access to your own workspace folder, where the output files go;
  - optionally, read access to table metadata, so declared primary and foreign keys are picked up. Without it the
    notebook works them out from the data.
- **Network:** the notebook installs `pydantic`, `sqlglot` and `pyyaml` from your workspace's configured Python
  package index (PyPI or a mirror). If you have no package access, tell us and we'll send those as wheels too.

## 3. Import the folder

Keep the folder structure as it is: the notebook finds `synth/` and `wheels/` (and, by default, `queries/`) next to
itself.

- **CLI:**
  ```bash
  databricks workspace import-dir ./customer_synth /Workspace/Users/<you>/customer_synth
  ```
- **UI:** in **Workspace → your folder → Import**, upload the `.zip` we sent.
- **Git folder:** add the repository as a Git folder.

## 4. Set the parameters

Open `01_profile_and_derive` and attach serverless compute or your cluster. Then **run the cells down to and
including the one that starts with `import os`**: the install cell (`%pip install ./wheels/...`), the restart, and
the settings cell. The settings cell only defines the parameters; it doesn't read any data. The parameters then
appear as widgets at the top of the notebook.

**Required.** The defaults are our test names, so change these:

| Widget | Set to |
|---|---|
| `source_catalog` | The catalog holding the source tables |
| `source_schema` | The schema holding the source tables |

**Usually left as they are:**

| Widget | Default | Meaning |
|---|---|---|
| `source_tables` | `@queries` | Profile exactly the tables the SQL uses. Other tables in the schema are ignored and never read. You can also give a comma list (`auths,merchants`), fully-qualified names from several schemas (`cat.s1.auths,cat.s2.merchants`), or a pattern (`auth_*`) |
| `queries_path` | `queries/` next to the notebook | Where your SQL workload is: a `.sql` file or a folder of them (subfolders included). Statements that don't parse are reported and skipped |
| `config_path` | `/Workspace/Users/<you>/customer_synth_runs/customer_3x_config.yaml` | Where the output files go. Keep the same path across reruns to resume |
| `sample_rows` | `20000000` | Size of the random row sample used for distributions. `0` reads every row |
| `key_sample_rows` | `200000000` | Size of the key sample used for skew and join statistics. `0` reads every row |
| `resume` | `yes` | Reuse the profiles an earlier run already saved (see step 6) |
| `top_n` | `100` | How many of the most frequent values to keep per column. Lower it to share fewer values |
| `copy_max_rows` | `1000` | Lookup tables this small are included row for row (see step 7). `0` includes none |
| `read_constraints` | `yes` | Read declared PK/FK constraints from Unity Catalog |
| `scale`, `date_mode`, `seed`, `grow_ids`, `target_*`, `source_prefix` | | Settings for the data we generate on our side. Leave them unless we ask you to change one |

## 5. Run it

Click **Run all**. The notebook:

1. Parses the SQL and lists the tables it will profile, with their column counts.
2. Profiles one table at a time and logs each step with timings. Each table's profile is saved as soon as that
   table finishes.
3. Works out how the tables relate (keys, joins, match rates), then writes the output files.
4. Shows tables you can review: the relationships, each table's settings, how each column will be generated, and
   any small lookup tables included in full.

**How it handles large tables.** To keep multi-terabyte tables affordable:
- **The full table is read only for simple aggregates:** row counts, null rates, distinct counts, min/max, and
  counts of the values your SQL filters on.
- **Distributions come from a random sample of rows.** That covers percentiles, frequent values and per-day counts.
- **Skew and join statistics come from a sample of key values.** Every row of a sampled key is kept, and the same
  keys are used on every table, so join match rates stay accurate.

Tables smaller than the sample sizes are read in full.

**Try one table first.** For a first run, set `source_tables` to your largest single table. The timings in the log
show what the full run will cost. Then set it back to `@queries`.

## 6. If a run stops partway

Run it again with the same `config_path` and `resume = yes`. Tables that already finished are loaded from their
saved profiles and aren't read again. A saved profile is only reused when it was made with the same settings and SQL
and the table's row count hasn't changed. Otherwise that table is profiled again.

## 7. Review the outputs before sending them

The files are under `config_path`:

| File | Send back? | Contents |
|---|---|---|
| `customer_3x_config.yaml` | **Yes** | The generation plan: table and column definitions (types, partitioning, keys), relationships between tables, a rule for every column, warnings, and the parsed SQL |
| `customer_3x_config_profiles.json` | **Yes** | Per-table and per-column statistics |
| `customer_3x_config_work/` | No | Working files (one profile per table, used for resume) |

**What the files contain about your data:**

- **Per column:**
  - row count, null rate, distinct count;
  - **minimum and maximum value**;
  - percentiles of numbers and dates;
  - per-day row counts for date columns;
  - string lengths;
  - **up to `top_n` of the most frequent values, with their counts**.
- **For mostly-unique text** (names, identifiers): only its format, meaning which characters appear at each
  position. Individual values are not kept, apart from the minimum and maximum above.
- **For filters in your SQL** (for example `country_cd = 'US'`): the literal values and how many rows match.
- **Code → description pairs**, where one column determines another (for example `category_cd` → `category_desc`).
- **Small lookup tables in full:** tables with at most `copy_max_rows` rows are included row for row under
  `copy_rows` in the YAML, and are also shown in the last cell of the notebook.
- **The text of the SQL queries.**

**How to share less:**

| Setting | Effect |
|---|---|
| `top_n` lower (e.g. `20`) | Fewer frequent values per column |
| `copy_max_rows = 0` | No lookup rows. We then generate those tables like any other, so their codes and descriptions become synthetic |
| Fewer queries in `queries_path` | Only the tables and columns those queries use are profiled in detail |

If any min/max values or frequent values can't be shared, tell us which columns **before** sending the files, and
we'll agree how to handle them.

## 8. Send the files back

Download `customer_3x_config.yaml` and `customer_3x_config_profiles.json` (**Workspace → file → Download**) and send them to
us through the channel we agreed.

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| `ModuleNotFoundError: synth` or `dbldatagen` | The folder structure changed, or the install cell didn't run. Keep `synth/` and `wheels/` next to the notebook, then run the install and restart cells again |
| The `%pip install ./wheels/...` cell can't find the wheel | Change the path in that cell to the wheel's full workspace path, e.g. `/Workspace/Users/<you>/customer_synth/wheels/dbldatagen-0.4.0.post2-py3-none-any.whl` |
| `queries_path ... doesn't exist` | Upload the `.sql` files to a `queries/` folder next to the notebook, or set `queries_path` (step 1) |
| `no source tables selected` | The table names in the SQL don't match the tables in `source_catalog.source_schema`. Check the schema, or list the tables in `source_tables` |
| `WARNING: the SQL uses ... table(s) not found` | Those tables live in another schema or catalog, or can't be read, so they aren't profiled. List all the benchmark tables in `source_tables` with fully-qualified names (`catalog.schema.table`, comma-separated) and run again |
| `WARNING: skipped ..., which can't be read` | A table or view named in the SQL can't be opened (a broken view, missing files, no permission). The run continues without it. Fix it or grant access if the benchmark needs it |
| Some statements reported as `could not parse` | They're skipped and the rest continue. Send us the list if they matter |
| A note that constraints couldn't be read | Not an error: keys are then worked out from the data. Set `read_constraints = no` to skip the lookup |
| `PERMISSION_DENIED` on a table | Grant `SELECT` on it, or drop it from `source_tables` |
| The run takes too long | Run fewer tables at a time with `source_tables` (resume keeps finished ones), or lower `sample_rows` / `key_sample_rows` |
| A failure partway through | Rerun with the same `config_path` and `resume = yes` (step 6). If it fails again on the same table, send us the error and the last log lines |
