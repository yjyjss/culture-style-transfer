# Culture Style Transfer

Research code for reference-edge-constrained style transfer and multi-scale control in ceramic product-image stylization.

## Repository scope

This archive contains all 12 Python source files present in the server experiment project at export, plus the local manuscript/figure/revision utilities and current LaTeX/BibTeX sources. It intentionally excludes credentials, model checkpoints, source images, generated images/PDFs, CSV result or rating files, machine environments and caches. The method diagram SVG is included as editable source.

The server scripts are preserved byte-for-byte. This is an archival research snapshot, not a claim that the experiments or manuscript are ready for publication. Source provenance and SHA-256 hashes are recorded in `CODE_MANIFEST.json`. No new experiment was run for this upload.

## Files

| File | Purpose |
|---|---|
| `stylize.py` | Frozen VGG encoder, Sobel prior, base objectives and optimization |
| `run_exp.py` | Paired base/edge-constrained experiment driver |
| `adain_exp.py` | AdaIN initialization and matched refinement controls |
| `wct_exp.py` | Local covariance-target reconstruction control |
| `attn_exp.py` | Local single-layer attention-target reconstruction control |
| `multiscale.py`, `multiscale_v2.py`, `multiscale_v3.py` | Historical and hybrid multi-scale objectives |
| `evaluate.py` | Saved-image VGG feature-distance diagnostic |
| `cfm.py` | Edge and relative Gram-distance diagnostics |
| `scan_lambda.py` | Additional structural-constraint coefficient runs |
| `analysis.py` | Rating-analysis utilities; real observations must be supplied separately |
| `tools/` | Figure assembly, descriptive summaries, revision and QA helpers |
| `manuscript/` | Current v3 LaTeX/BibTeX and editable method diagram source |

## Experiment setup

Install compatible versions of the dependencies in `requirements.txt` in an isolated environment. The original scripts select CUDA when available. The requirements list is not a frozen reproduction environment.

The scripts expect `data/content/`, `data/style/`, and, for the AdaIN branch, `models/decoder.pth`. These assets are not included. Verify source-image rights and checkpoint provenance before reuse.

Example entry point, once the assets and environment have been prepared:

```bash
python run_exp.py --content_dir data/content --style_dir data/style --out_dir output
```

Review scripts before execution: several research drivers run at module import or write into fixed output directories. Preserve previous outputs and use an isolated project copy for new runs.

## Known implementation and evidence boundaries

- The archived image loader and encoder both perform ImageNet normalization. Correcting this requires a separately versioned paired replication; archived numbers should not be relabeled as standard-preprocessing results.
- `evaluate.py` parses the content name before the first underscore; the saved evaluation covered only the tea-set carrier. Fix and validate this before treating results as complete coverage.
- The local covariance and attention controls are not complete reproductions of the original published WCT/AdaAttN systems. The AdaIN checkpoint source and encoder compatibility also require verification.
- Edge diagnostics share their prior with the optimized loss. They are not independent cultural-semantic validation. The VGG-distance diagnostic is not standard LPIPS; FID was not computed.
- The available main collection has five references and three ceramic carriers. Multi-scale intervention outputs cover one pair. This does not establish general cultural or product-domain performance.
- Prepared instruments or simulated ratings are not empirical human evidence. No human-rating data are included here.
- The requirements do not specify the original complete environment, per-run completion flags, or measured runtimes.

## Manuscript and figure utilities

The optional utilities preserve the research workspace conventions, including local filesystem paths and dependencies on installed Codex skill audit scripts. They are provenance and editing utilities, not portable end-user CLIs. Adjust their paths before use. `tools/requirements.txt` lists their additional Python imports. Manuscript image PDFs and other figure assets are excluded, so the LaTeX source alone is not a self-contained compilation bundle.

## Rights

No open-source license is assigned by this upload. Confirm authorship and third-party code/checkpoint permissions before making the repository public or selecting a license. Repository access does not establish rights to redistribute the excluded images or model assets.
