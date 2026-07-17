# Getting Started

StructureZyme is a modular pipeline for enzyme structure/function prediction.
This is a stub guide covering one-time model-weight setup; Phase E will expand it.

## Catalytic-residue model weights (Squidly)

StructureZyme does **not** manage model weights itself. Catalytic-residue
prediction shells out to the `squidly` CLI, which must be available on your
`$PATH`. The `squidly` package ships and downloads its own model ensemble from
the HuggingFace repo `WillRieger/Squidly`; StructureZyme never bundles or loads
`.pth` files directly.

Run the following once after installing `squidly`:

```bash
# Weights are shipped and downloaded by the `squidly` package itself
# (HuggingFace repo WillRieger/Squidly). Run once after installing squidly:
pip install squidly
python -c "import squidly, os; os.system(f'python {os.path.dirname(squidly.__file__)}/download_models_hf.py')"

# Verify the ensemble landed in site-packages/squidly/models/{3B,15B}/:
python -c "import squidly, os, glob; d=os.path.join(os.path.dirname(squidly.__file__),'models'); print(sorted(glob.glob(d+'/*/*.p*')))"
```

Notes:

- StructureZyme requires the `squidly` CLI on `$PATH` and does not manage
  weights itself. If the CLI is missing, the Squidly step fails loudly rather
  than silently falling back to local weights.
- ESM2 inference (used by Squidly) needs a GPU.
