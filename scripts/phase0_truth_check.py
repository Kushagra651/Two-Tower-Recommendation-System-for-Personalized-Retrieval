import json, os
import pandas as pd

P = "data/processed"
raw = pd.read_csv(f"{P}/merged_interactions.csv")
proc = pd.read_csv(f"{P}/processed_interactions.csv")
train = pd.read_csv(f"{P}/train.csv")
val = pd.read_csv(f"{P}/val.csv")
test = pd.read_csv(f"{P}/test.csv")
maps = json.load(open(f"{P}/id_maps.json"))

print("raw rows (all ratings):      ", len(raw))
for t in (1, 2, 3, 4, 5):
    print(f"  raw rows with rating >= {t}: ", int((raw.Rating >= t).sum()))
print("processed_interactions rows: ", len(proc))
print("train / val / test rows:     ", len(train), len(val), len(test))
print("num_users / num_items (maps):", maps["num_users"], maps["num_items"])
print("raw unique users / items:    ", raw.UserID.nunique(), raw.MovieID.nunique())
print("eval users (test rows):      ", test.user_idx.nunique())
print("duplicate (user,item) in proc:", int(proc.duplicated(["user_idx", "item_idx"]).sum()))
print("users with <3 interactions in proc:",
      int((proc.groupby("user_idx").size() < 3).sum()))

