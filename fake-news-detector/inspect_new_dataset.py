import json

path = "dataset_binary_checkpoint_868.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("TYPE:", type(data).__name__)
print("ROWS:", len(data))

if isinstance(data, list):
    print("FIELDS:", list(data[0].keys()))

    for i, row in enumerate(data[:3], 1):
        print(f"\n--- ROW {i} ---")
        print(json.dumps(row, ensure_ascii=False, indent=2))
