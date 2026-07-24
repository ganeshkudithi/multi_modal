import csv

with open("matclip_phase1_cluster.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

print("Header columns:", len(header))
for i, h in enumerate(header):
    print(i, h)
