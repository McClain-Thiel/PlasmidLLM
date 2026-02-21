# Vocabulary Update: v1 → v3

## Summary

Updated from `token_vocabulary.json` (v1) to `token_vocabulary_v3.json` as the primary vocabulary.

## Changes

### Vocabulary Size
- **v1**: 101 tokens
- **v3**: 250 tokens ⭐

### Token Coverage

**v1** includes:
- Control tokens: `<BOS>`, `<EOS>`, `<SEP>`, `<PAD>`, `<UNK>`
- Copy number: `<COPY_HIGH>`, `<COPY_LOW>`, `<COPY_UNK>`
- AMR (15 tokens): `<AMR_KANAMYCIN>`, `<AMR_AMPICILLIN>`, etc.
- Origins (20 tokens): `<ORI_COLE1>`, `<ORI_F1>`, etc.
- Promoters (21 tokens): `<PROM_CMV>`, `<PROM_T7>`, etc.
- Elements (31 tokens): `<ELEM_IRES>`, `<ELEM_POLYA_SV40>`, etc.
- Vectors (12 tokens): `<VEC_MAMMALIAN>`, `<VEC_BACTERIAL>`, etc.

**v3** includes all of the above PLUS:
- **149 additional `<FEAT_*>` tokens** for fine-grained feature annotation
  - Examples: `<FEAT_3XFLAG>`, `<FEAT_6XHIS>`, `<FEAT_AMPR>`, `<FEAT_T7_PROMOTER>`, etc.
  - These provide more detailed annotations beyond the high-level categories

### File Structure

**v1** (`token_vocabulary.json`):
```json
{
  "version": "1.0",
  "total_tokens": 101,
  "token_to_id": {"<BOS>": 0, ...},
  "id_to_token": {"0": "<BOS>", ...},
  "categories": {...},
  ...
}
```

**v3** (`token_vocabulary_v3.json`):
```json
{
  "<BOS>": 0,
  "<EOS>": 1,
  "<SEP>": 2,
  ...
}
```

v3 is a simpler format - just a direct token → ID mapping.

## Impact

### ✅ Benefits
1. **More comprehensive**: 250 tokens vs 101 tokens
2. **Better feature coverage**: Extensive `<FEAT_*>` tokens for detailed annotations
3. **Cleaner format**: Simple dict structure
4. **Standard separator**: Uses `<SEP>` consistently

### ⚠️ Issues to Address

1. **Training data uses `<SEQ>` instead of `<SEP>`**
   - The vocabulary uses `<SEP>` as the sequence separator
   - Training data uses `<SEQ>` which doesn't exist in v3
   - Will be converted to `<UNK>` during tokenization
   - **Fix**: Regenerate training data to replace `<SEQ>` with `<SEP>`

2. **Species tokens missing**
   - Training data uses `<SP_RAT>` which is not in vocabulary
   - Neither v1 nor v3 have species tokens
   - **Fix**: Either add to vocabulary OR remove from training data

## Files Updated

✅ `data/special_tokens.txt` - Regenerated from v3 (now 250 tokens)
✅ `data/README.md` - Updated documentation
✅ `README.md` - Updated project structure
✅ `TEST_RESULTS.md` - Updated test findings
✅ Smoke tests passing with v3 vocabulary

## Tokenizer Stats

With v3 vocabulary + DNA bases:
- **250** special tokens from vocabulary
- **10** DNA base tokens (A, T, C, G, N + lowercase)
- **260 total tokens** in tokenizer

## Next Steps

### For Development (Sample Data)
Current setup works but has some `<UNK>` conversions:
```bash
# Smoke tests pass
PYTHONPATH=src:$PYTHONPATH python tests/smoke_test.py
```

### For Production Training
Before training on full dataset:
1. **Validate tokens**: Ensure all tokens in training data exist in v3 vocabulary
2. **Replace `<SEQ>` → `<SEP>`**: Update training data to use correct separator
3. **Handle species tokens**: Decide whether to add to vocab or remove from data

### Validation Script (Recommended)

Create a script to check token coverage:
```python
import json
import pyarrow.parquet as pq
import re

# Load vocabulary
with open('data/token_vocabulary_v3.json') as f:
    vocab = set(json.load(f).keys())

# Load training data
table = pq.read_table('data/training_pairs.parquet')

# Extract all tokens from all sequences
all_tokens = set()
for text in table.column('full_text'):
    tokens = re.findall(r'<[^>]+>', text.as_py())
    all_tokens.update(tokens)

# Find missing tokens
missing = all_tokens - vocab
if missing:
    print(f"❌ {len(missing)} tokens in data but NOT in vocabulary:")
    for token in sorted(missing):
        print(f"  {token}")
else:
    print("✅ All tokens in training data exist in vocabulary!")
```

## Recommendation

✅ **Use v3** (`token_vocabulary_v3.json`) as the primary vocabulary:
- It's more comprehensive (250 vs 101 tokens)
- Includes detailed feature annotations
- Simpler format
- This is now the default in all configs and examples
