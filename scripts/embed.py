"""Embed every English and NULL-language work with SPECTER2 and store the vectors in the embeddings table.

The input for each work follows the decisions in notebooks/test_specter2_embedding.ipynb: title [SEP] abstract,
truncated to 512 tokens. A junk abstract (under 50 characters, or the same text on 5 or more works) is left out,
so the work is embedded from its title alone.

Resumable: works already in embeddings are skipped, so after a crash or Ctrl+C, run it again.
It writes to the database, so close any notebook connection to it first.

Run from the repo root:  .venv\\Scripts\\python scripts\\embed.py
"""
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from ingest import corpus

BATCH_SIZE = 16      # fastest in the notebook's timing test
CHUNK_SIZE = 10_000  # works per database write; a crash loses at most one chunk (a minute or two)


def load_model():
    """SPECTER2: the specter2_base model plus the proximity adapter, on the GPU if torch can see one."""
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def embed(texts, tokenizer, model):
    """One 768-number vector per text: the final hidden state of the [CLS] token."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512,
                       return_tensors="pt", return_token_type_ids=False).to(model.device)
    with torch.no_grad():  # inference only, so skip the bookkeeping needed for training
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].cpu().numpy()


def create_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                    work_id VARCHAR,
                    embedding FLOAT[768],
                    PRIMARY KEY (work_id)
                    )""")


def create_todo(con):
    """Temp table of the works still to embed and their input text, numbered longest first.

    Longest first keeps texts of similar length in the same batch (less padding), and the
    longest batches, the ones most likely to run out of GPU memory, come in the first minute.
    """
    con.execute("""
        CREATE TEMP TABLE todo AS
        WITH repeated AS (  -- boilerplate: the same abstract text on 5 or more works
            SELECT trim(abstract) AS text
            FROM works
            WHERE abstract IS NOT NULL
            GROUP BY 1
            HAVING count(*) >= 5
        ),
        inputs AS (
            SELECT work_id, title,
                   CASE WHEN length(trim(coalesce(abstract, ''))) < 50
                             OR trim(abstract) IN (SELECT text FROM repeated)
                        THEN ''  -- junk abstract: embed the title alone
                        ELSE abstract
                   END AS abstract
            FROM works
            WHERE (language = 'en' OR language IS NULL)
              AND work_id NOT IN (SELECT work_id FROM embeddings)
        )
        SELECT row_number() OVER (ORDER BY length(title) + length(abstract) DESC) AS position, *
        FROM inputs
        WHERE trim(title) <> '' OR abstract <> ''  -- an empty title with a junk abstract leaves nothing to embed
        ORDER BY position
    """)
    return con.sql("SELECT count(*) FROM todo").fetchone()[0]


def main(db_path):
    tokenizer, model = load_model()
    if model.device.type != "cuda":
        sys.exit("torch cannot see the GPU. On the CPU this run would take about a week.")
    print("SPECTER2 loaded on", torch.cuda.get_device_name(0))

    with duckdb.connect(db_path) as con:
        create_tables(con)
        n_done = con.sql("SELECT count(*) FROM embeddings").fetchone()[0]
        print(f"{n_done:,} works already embedded. Finding the rest (about a minute)...")
        n_todo = create_todo(con)
        print(f"{n_todo:,} works to embed, longest first. The time estimate starts high and falls as texts get shorter.")

        start = time.perf_counter()
        for first in range(1, n_todo + 1, CHUNK_SIZE):
            chunk = con.execute("SELECT work_id, title, abstract FROM todo WHERE position BETWEEN ? AND ? ORDER BY position",
                                [first, first + CHUNK_SIZE - 1]).fetchall()
            texts = [title + tokenizer.sep_token + abstract for _, title, abstract in chunk]
            vectors = np.concatenate([embed(texts[i:i + BATCH_SIZE], tokenizer, model)
                                      for i in range(0, len(texts), BATCH_SIZE)])

            rows = pd.DataFrame({"work_id": [work_id for work_id, _, _ in chunk], "embedding": list(vectors)})
            con.execute("INSERT INTO embeddings SELECT work_id, embedding::FLOAT[768] FROM rows")  # commits this chunk

            n_embedded = first - 1 + len(chunk)
            rate = n_embedded / (time.perf_counter() - start)
            print(f"{n_embedded:,} / {n_todo:,} ({n_embedded / n_todo:.1%}), {rate:.0f} works/s, "
                  f"about {(n_todo - n_embedded) / rate / 3600:.1f} hours left", flush=True)

        print(f"Finished: {con.sql('SELECT count(*) FROM embeddings').fetchone()[0]:,} works embedded in total.")


if __name__ == "__main__":
    main(corpus["db_path"])
