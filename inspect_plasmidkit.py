import plasmidkit as pk
import sys
import os

# Find a suitable test file
test_dir = "tests/data"
test_file = None
for f in os.listdir(test_dir):
    if f.endswith(".fasta") or f.endswith(".gbk") or f.endswith(".gb"):
        test_file = os.path.join(test_dir, f)
        break

if not test_file:
    print("No test file found in tests/data")
    sys.exit(0)

print(f"Testing on {test_file}")

try:
    record = pk.load_record(test_file)
    # Use skip_prodigal=True as recommended for speed/ORI focus
    annotations = pk.annotate(record, skip_prodigal=True)

    print(f"Found {len(annotations)} features.")
    
    types = set()
    origins = []
    markers = []
    
    for ann in annotations:
        # Check available attributes on the annotation object
        # The user example showed .type, .id, .start, .end
        feat_type = getattr(ann, 'type', 'unknown')
        feat_id = getattr(ann, 'id', 'unknown')
        
        types.add(feat_type)
        
        if feat_type in ['rep_origin', 'origin', 'ori']:
            origins.append(feat_id)
        if feat_type in ['marker', 'resistance', 'cds_resistance']: # Guessing marker types
            markers.append(feat_id)
            
    print(f"Feature Types Found: {types}")
    print(f"Origins Found: {origins}")
    print(f"Markers Found: {markers}")

    # Also check if we can access the internal registry/database mentioned
    # "thats json is installed with the library"
    # Let's inspect the package location or exposed attributes
    import pkg_resources
    try:
        dist = pkg_resources.get_distribution("plasmidkit")
        print(f"PlasmidKit location: {dist.location}")
    except:
        pass

except Exception as e:
    print(f"Error: {e}")
