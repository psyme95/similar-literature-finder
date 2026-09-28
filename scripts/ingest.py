# Imports
import math
import os
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

# Settings
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load config file and store
def get_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

config = get_config(REPO_ROOT / "config" / "config.yaml")
corpus = config["corpus"]

load_dotenv(REPO_ROOT / ".env")
config["api_key"] = os.environ["OPENALEX_API_KEY"]

# Setup parameters for API query
works_url = "https://api.openalex.org/works"
subfields = "|".join(str(subfield) for subfield in corpus["subfields"])
params = {
    "select": ",".join(corpus["cols_to_select"]),
    "api_key": config["api_key"],
}

# Functions
def paginate(url, params=None, per_page=200):
    """Yield successive pages from a cursor-paginated OpenAlex endpoint."""
    params = dict(params or {})
    params["per-page"] = per_page
    cursor = "*"

    while cursor:
        params["cursor"] = cursor
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()

        if not page.get("results"):
            break

        yield page
        cursor = page.get("meta", {}).get("next_cursor")

def reconstruct_abstract(inverted_abstract):
    if not inverted_abstract:
        return None, 0

    abstract = []
    for word, positions in inverted_abstract.items():
        for position in positions:
            while len(abstract) <= position:
                abstract.append(None)
            abstract[position] = word

    n_gaps = abstract.count(None)
    text = " ".join([word for word in abstract if word is not None])

    return text, n_gaps

def remove_prefix(item):
    if item is None:
        return None

    value = item.removeprefix("https://doi.org/")
    value = value.removeprefix("https://openalex.org/subfields/")
    value = value.removeprefix("https://openalex.org/")

    return value

def flatten_work(work):
    work_id = remove_prefix(work['id'])
    abstract, gap_count = reconstruct_abstract(work['abstract_inverted_index'])

    work_row = {'work_id': work_id,
                'doi': remove_prefix(work['doi']),
                'title': work['title'],
                'abstract': abstract,
                'abstract_gap_count': gap_count,
                'publication_year': work['publication_year'],
                'publication_date': work['publication_date'],
                'cited_by_count': work['cited_by_count'],
                'subfield_id': int(remove_prefix(work['primary_topic']['subfield']['id'])),
                'subfield_name': work['primary_topic']['subfield']['display_name'],
                'topic_id': remove_prefix(work['primary_topic']['id']),
                'topic_name': work['primary_topic']['display_name'],
                'language': work['language']
                }

    author_rows = []
    for author_order, authorship in enumerate(work['authorships'], start=1):
        author_row = {'work_id': work_id,
                      'author_order': author_order,
                      'author_id': remove_prefix(authorship['author']['id']),
                      'author_name': authorship['author']['display_name']
                      }
        author_rows.append(author_row)

    reference_rows = []
    for url in work['referenced_works']:
        reference_row = {'citing_work_id': work_id, 'cited_work_id': remove_prefix(url)}
        reference_rows.append(reference_row)

    return (work_row, author_rows, reference_rows)

def year_params(year):
    new_params = params.copy()
    new_params['filter'] = ",".join([
        f"publication_year:{year}",
        f"has_abstract:{str(corpus['has_abstract']).lower()}",
        f"primary_topic.subfield.id:{subfields}",
    ])
    return new_params

def insert_rows(con, table, rows):
    if not rows:
        print(f"No rows for {table} on this page, skipping.")
        return

    rows_df = pd.DataFrame(rows)
    con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM rows_df")


def ingest_year(con, year):
    con.begin()
    try:
        pages = paginate(works_url, year_params(year), per_page=corpus['per_page'])
        for page_number, page in enumerate(pages, start=1):
            if page_number == 1:
                n_works = page["meta"]["count"]
                n_pages = math.ceil(n_works / corpus["per_page"])
                print(f"{n_works} works in {year}, {n_pages} pages to fetch.")

            works_list = []
            authors_list = []
            references_list = []

            for work in page['results']:
                work_row, work_authors, work_references = flatten_work(work)
                
                works_list.append(work_row)
                authors_list.extend(work_authors)
                references_list.extend(work_references)
                
            insert_rows(con, "works", works_list)
            insert_rows(con, "work_authors", authors_list)
            insert_rows(con, "work_references", references_list)

            if page_number % 50 == 0: print(page_number, "pages parsed.")
        con.commit()
        print("Finished page parsing.")

    except BaseException:
        con.rollback()
        raise

def create_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS works (
                    work_id VARCHAR,
                    doi VARCHAR,
                    title VARCHAR,
                    abstract VARCHAR,
                    abstract_gap_count INTEGER,
                    publication_year INTEGER,
                    publication_date DATE,
                    cited_by_count INTEGER,
                    subfield_id INTEGER,
                    subfield_name VARCHAR,
                    topic_id VARCHAR,
                    topic_name VARCHAR,
                    language VARCHAR,
                    PRIMARY KEY (work_id)
                    )""")

    con.execute("""CREATE TABLE IF NOT EXISTS work_authors (
                    work_id VARCHAR,
                    author_order INTEGER,
                    author_id VARCHAR,
                    author_name VARCHAR,
                    PRIMARY KEY (work_id, author_order)
                    )""")

    con.execute("""CREATE TABLE IF NOT EXISTS work_references (
                    citing_work_id VARCHAR,
                    cited_work_id VARCHAR
                    )""")

    
if __name__ == "__main__":
    with duckdb.connect(corpus["db_path"]) as con:
        create_tables(con)
        years = con.sql("SELECT DISTINCT publication_year FROM works").fetchall()
        done_years = {row[0] for row in years}

        for year in range(corpus['min_year'], 2027):
            if year in done_years:
                print(year, "already present in database, skipping")
                continue

            # One cheap request (1 credit): how many pages this year needs, and how many credits are left today
            check_params = year_params(year)
            check_params["per-page"] = 1
            response = requests.get(works_url, params=check_params, timeout=30)
            response.raise_for_status()
            pages_needed = math.ceil(response.json()["meta"]["count"] / corpus["per_page"])
            credits_left = int(response.headers["X-RateLimit-Remaining"])

            if credits_left < pages_needed + 10:  # small margin in case the count grows during the year
                print(f"Stopping before {year}: it needs {pages_needed} credits and {credits_left} are left today.")
                print("Run the script again after the daily budget resets.")
                break

            ingest_year(con, year)