# Literature search tool

A personal tool for finding relevant papers. It holds a corpus of OpenAlex works in DuckDB, to be searched with SPECTER2 embeddings combined with keyword matching. 

## Corpus

- **Source:** [OpenAlex](https://openalex.org) (CC0), through its API.
- **Filters** (in [config/config.yaml](config/config.yaml)): the work's primary topic is in one of six subfields (Ecological Modeling; Ecology; Nature and Landscape Conservation; Ecology, Evolution, Behavior and Systematics; Modeling and Simulation; Statistics and Probability), it was published in 2010 or later, and it has an abstract. That gives about 2.65 million works.
- **Storage:** a DuckDB file with three tables:
  - `works`: one row per work, with the abstract rebuilt from OpenAlex's inverted index
  - `work_authors`: one row per author per work
  - `work_references`: one row per citation

## Setup

Python 3.13 on Windows.

```
python -m venv .venv
.venv\Scripts\python -m pip install duckdb pandas requests pyyaml python-dotenv jupyter
```

Create a `.env` file in the repo root containing your OpenAlex API key:

```
OPENALEX_API_KEY=your-key
```

## Building the corpus

Run this from the repo root:

```
.venv\Scripts\python scripts\ingest.py
```

- It downloads one publication year at a time, each in its own database transaction, so a failure never leaves a half-loaded year.
- It is safe to run again: years already in the database are skipped.
- An API key has a daily budget of 10,000 requests, and the full corpus needs about 13,300 pages of 200 works. The ingest therefore takes two runs on consecutive days. The script stops by itself before any year the remaining budget can't cover.

## Checking the corpus

[notebooks/get_openalex_corpus.ipynb](notebooks/get_openalex_corpus.ipynb) holds sanity checks on the API data and checks on the loaded database. It opens the database read-only. Close its connection before running the ingest, because DuckDB allows only one process to write at a time.

## Layout

| Path | Contents |
|---|---|
| `config/config.yaml` | corpus filters, fields fetched, page size, database path |
| `scripts/ingest.py` | the ingest |
| `notebooks/` | exploration and checks |
| `docs/build-plan.md` | stages, checkpoints and scope boundaries |
| `data/` | the DuckDB file (not in git) |
