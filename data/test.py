import pandas as pd

# Lade die erste Chunk-Datei deines Datensatzes
file_path = "/home/mm/dev/il-visual-tactile/data/lerobot-datasets/user/foil_lifting_baseline/data/chunk-000/file-000.parquet"
df = pd.read_parquet(file_path)

# Betrachte den exakten Inhalt des ersten Frames
print("State:", df["observation.state"].iloc[0])
print("Action:", df["action"].iloc[0])
