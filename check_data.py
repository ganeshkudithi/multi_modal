import csv

csv_path = "matclip_phase1_cluster.csv"

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    for i, row in enumerate(reader, start=1):
        if i in [1, 52]:
            print(f"Line {i}: {len(row)} columns")
            for j, col in enumerate(row):
                print(f"{j:02d}: {repr(col)}")
