"""
src/data/build_movie_lookup.py
---------------------------------
Builds a small movies.csv (MovieID, Title, Genres) from the merged
interactions table, so the serving layer doesn't need to load the full
multi-million-row interactions CSV just to look up a title.

Usage:
    python -m src.data.build_movie_lookup \
        --in_path data/processed/merged_interactions.csv \
        --out_path data/processed/movies.csv
"""

import argparse
import os

import pandas as pd


def main(in_path: str, out_path: str):
    df = pd.read_csv(in_path, usecols=["MovieID", "Title", "Genres"])
    movies = df.drop_duplicates(subset="MovieID").sort_values("MovieID").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    movies.to_csv(out_path, index=False)
    print(f"Saved {len(movies)} unique movies -> {out_path}")
    return movies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a MovieID -> Title/Genres lookup table")
    parser.add_argument("--in_path", type=str, default="data/processed/merged_interactions.csv")
    parser.add_argument("--out_path", type=str, default="data/processed/movies.csv")
    args = parser.parse_args()

    main(args.in_path, args.out_path)