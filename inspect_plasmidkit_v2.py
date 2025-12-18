import plasmidkit as pk
import inspect
import sys
import os

print("Inspecting pk.annotate signature:")
try:
    print(inspect.signature(pk.annotate))
except Exception as e:
    print(f"Could not get signature: {e}")

test_file = "tests/data/GenBank AAEN01000030.1.gbk"
if not os.path.exists(test_file):
    # Try to find any file
    for f in os.listdir("tests/data"):
         if f.endswith(".gbk") or f.endswith(".fasta"):
             test_file = os.path.join("tests/data", f)
             break

print(f"Testing on {test_file}")

try:
    record = pk.load_record(test_file)
    # Try without the failing argument
    print("Running annotate()...")
    annotations = pk.annotate(record)

    print(f"Found {len(annotations)} features.")
    
    types = set()
    origins = []
    
    for ann in annotations:
        # Check attributes again
        feat_type = getattr(ann, 'type', 'unknown')
        feat_id = getattr(ann, 'id', 'unknown')
        
        types.add(feat_type)
        
        if feat_type in ['rep_origin', 'origin', 'ori']:
            origins.append(feat_id)
            
    print(f"Feature Types Found: {types}")
    print(f"Origins Found: {origins}")

except Exception as e:
    print(f"Error running annotate: {e}")
