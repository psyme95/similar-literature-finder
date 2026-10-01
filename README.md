# Literature search tool

A personal tool for finding relevant papers. It holds a corpus of OpenAlex works in DuckDB, to be searched with SPECTER2 embeddings combined with keyword matching.

## Corpus

- **Source:** [OpenAlex](https://openalex.org) (CC0), through its API.
- **Filters** (in [config/config.yaml](config/config.yaml)): the work's primary topic is in one of six subfields (Ecological Modeling; Ecology; Nature and Landscape Conservation; Ecology, Evolution, Behavior and Systematics; Modeling and Simulation; Statistics and Probability), it was published in 2010 or later, and it has an abstract. Works with no title are left out: they are supplementary-file and journal-issue records rather than papers.
- **Storage:** a DuckDB file with three tables, plus a fourth once the embeddings are built:
  - `works`: one row per work, with the abstract rebuilt from OpenAlex's inverted index
  - `work_authors`: one row per author per work
  - `work_references`: one row per citation
  - `embeddings`: one 768-number SPECTER2 vector per embedded work

## Setup

Python 3.13 on Windows. Embedding needs an NVIDIA GPU.

```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install duckdb pandas requests pyyaml python-dotenv jupyter transformers adapters scikit-learn matplotlib
```

- Install torch first, from PyTorch's CUDA index. On Windows the torch on PyPI is CPU-only, and pip won't replace an installed CPU build unless you add `--force-reinstall`. The `cu130` build needs an NVIDIA driver that supports CUDA 13; [pytorch.org](https://pytorch.org/get-started/locally/) gives the command for other setups.
- The SPECTER2 model (under 1 GB) downloads from Hugging Face the first time it is used.

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
- A work already in the database is not stored twice. OpenAlex sometimes moves a work to another publication year after it was stored; the copy already held is kept.
- Works with no title are skipped.
- An API key has a daily budget of 10,000 requests, and the full corpus needs about 13,300 pages of 200 works. The ingest therefore takes two runs on consecutive days. The script stops by itself before any year the remaining budget can't cover.

## Embedding the corpus

Run this from the repo root:

```
.venv\Scripts\python scripts\embed.py
```

- **Model:** SPECTER2, which is `allenai/specter2_base` with the `allenai/specter2` proximity adapter, loaded with the `adapters` library. Each work becomes one vector: the final hidden state of the `[CLS]` token.
- **Which works:** those in English or with no language recorded. With the skips below, that is 2,270,167 works.
- **Input:** the title and abstract joined by the tokenizer's separator token, truncated to 512 tokens (about 6% of inputs are cut short). A junk abstract is left out, so the work is embedded from its title alone. Junk means under 50 characters, or text shared by 5 or more works, such as the boilerplate on 68,658 IUCN Red List entries. Works with an empty title and a junk abstract have nothing to embed and are skipped.
- **Time:** about 5 hours on an RTX 2080 Ti. On a CPU it would take about a week, so the script stops if torch can't see a GPU.
- **Resumable:** it saves every 10,000 works and skips works already embedded, so after a crash or Ctrl+C, run it again.

The checks behind these choices are in [notebooks/test_specter2_embedding.ipynb](notebooks/test_specter2_embedding.ipynb).

## Notebooks

- [notebooks/get_openalex_corpus.ipynb](notebooks/get_openalex_corpus.ipynb): sanity checks on the API data and checks on the loaded database.
- [notebooks/test_specter2_embedding.ipynb](notebooks/test_specter2_embedding.ipynb): SPECTER2 on a handful of known papers; the corpus checks that set the embedding input (token lengths, languages, titles, junk abstracts); embedding speed on the CPU and the GPU.

Both open the database read-only. DuckDB won't let a script open the file for writing while any other process has it open, so close the notebooks' connections before running either script. If the file is still locked after `con.close()`, restart the kernel.

## Layout

| Path | Contents |
|---|---|
| `config/config.yaml` | corpus filters, fields fetched, page size, database path |
| `scripts/ingest.py` | the ingest |
| `scripts/embed.py` | the embedding run |
| `notebooks/` | exploration and checks |
| `docs/build-plan.md` | stages, checkpoints and scope boundaries |
| `data/` | the DuckDB file (not in git) |
