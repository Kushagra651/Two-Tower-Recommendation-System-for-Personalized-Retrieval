"""
merge_ml1m.py
-------------
Loads the three raw MovieLens-1M files (movies.dat, ratings.dat, users.dat),
merges them into a single interactions table, and saves the result to
data/processed/ for EDA and downstream preprocessing.

Raw file formats (all '::' delimited, no header):
    movies.dat  -> MovieID::Title::Genres
    ratings.dat -> UserID::MovieID::Rating::Timestamp
    users.dat   -> UserID::Gender::Age::Occupation::Zip-code

Usage:
    python merge_ml1m.py --raw_dir data/raw --out_dir data/processed
"""

import argparse
import os

import pandas as pd


def load_movies(path: str) -> pd.DataFrame:
    """Load movies.dat -> MovieID, Title, Genres"""
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",       # ml-1m titles contain non-UTF8 chars
        header=None,
        names=["MovieID", "Title", "Genres"],
    )
    return movies


def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings.dat -> UserID, MovieID, Rating, Timestamp"""
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )
    return ratings


def load_users(path: str) -> pd.DataFrame:
    """Load users.dat -> UserID, Gender, Age, Occupation, Zip"""
    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip"],
        dtype={"Zip": str},        # keep zip codes like '02460' intact
    )
    return users


def merge_all(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Core interaction table is ratings.dat. We left-join movies and users
    onto it (left join keeps every rating row even if metadata is missing).
    """
    df = ratings.merge(movies, on="MovieID", how="left")
    df = df.merge(users, on="UserID", how="left")

    # Friendly datetime column from unix timestamp (useful for EDA + time-split)
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")

    return df


def main(raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    movies = load_movies(os.path.join(raw_dir, "movies.dat"))
    ratings = load_ratings(os.path.join(raw_dir, "ratings.dat"))
    users = load_users(os.path.join(raw_dir, "users.dat"))

    print(f"movies.dat  -> {movies.shape[0]} rows")
    print(f"ratings.dat -> {ratings.shape[0]} rows")
    print(f"users.dat   -> {users.shape[0]} rows")

    merged = merge_all(ratings, movies, users)
    print(f"merged      -> {merged.shape[0]} rows, {merged.shape[1]} columns")

    # sanity check: row count after merge should equal ratings row count
    assert merged.shape[0] == ratings.shape[0], "Merge changed row count — check join keys!"

    out_path = os.path.join(out_dir, "merged_interactions.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved merged dataset to: {out_path}")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge MovieLens-1M raw files")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    main(args.raw_dir, args.out_dir)