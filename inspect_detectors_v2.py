import plasmidkit as pk
import os

test_file = "tests/data/GenBank AAEN01000030.1.gbk"
if not os.path.exists(test_file):
     for f in os.listdir("tests/data"):
         if f.endswith(".gbk") or f.endswith(".fasta"):
             test_file = os.path.join("tests/data", f)
             break

print(f"Testing detectors on {test_file}")

try:
    print("Running annotate(detectors=['ori'])...")
    annotations = pk.annotate(test_file, detectors=["ori"])
    print(f"Found {len(annotations)} features.")
    print(f"Types: {set(a.type for a in annotations)}")
    
except Exception as e:
    print(f"Error: {e}")
