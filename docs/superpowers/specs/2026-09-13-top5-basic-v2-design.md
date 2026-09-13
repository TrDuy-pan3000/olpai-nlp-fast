# Top-5 Basic V2 Design

## Goal

Replace the tiny standard `nn.Transformer` with a readable implementation derived from the actual `external/btc-drive/Training.ipynb`, while remaining practical on a Colab T4.

## Scope

- Keep: joint 8k SentencePiece BPE, `<2vi>` direction token, length filtering, RMSNorm, RoPE, grouped-query attention, SwiGLU, pre-norm encoder/decoder, tied embeddings, label smoothing, AdamW, inverse-square-root warmup, AMP, gradient clipping, BLEU, checkpoints, batched greedy decode, optional beam-3, tqdm progress, and submission packaging.
- Remove: contrastive stages, projection heads, span masking, back-translation, R-Drop, SAM, and Vietnamese-to-Chinese scheduling.
- Fix: do not reproduce the source notebook's mislabeled `vi2zh` samples that fail to swap source and target.

## Contest Configuration

- Default mode is full training, not smoke testing.
- Full model: vocab 8,000; d_model 256; 8 query heads; 4 KV heads; 4 encoder layers; 4 decoder layers; SwiGLU width 768; max length 40; batch 128; at most 15 epochs.
- Smoke mode remains available as an explicit opt-in for debugging.
- The notebook pulls the latest `main` branch before each run and prints a conspicuous mode banner.

## Data Flow

Download dataset -> normalize and deduplicate -> group-aware train/validation split -> train joint tokenizer -> remove over-length pairs -> construct direction-tokenized batches -> train -> choose checkpoint by validation BLEU -> preview translations -> generate public/private CSV files -> package ZIP.

## Verification

Unit tests cover RMSNorm scale, RoPE/GQA tensor shapes, padding and causal masking, direction tokens, length filtering, full model forward pass, weight tying, notebook full-mode defaults, and the existing data/submission utilities. A Colab run must show full-data counts, an approximately 10M+ parameter model, multiple epochs, and nontrivial predictions.
