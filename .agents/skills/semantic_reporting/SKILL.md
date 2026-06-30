---
name: semantic_reporting
description: Guidelines for plotting graphs, handling RTL markdown docs, and managing experiment files in the Semantic Communication project.
---

# Semantic Reporting & Graphing Skill

When working on this project (Semantic Communication Hybrid Compressor), you MUST follow these specific instructions regarding outputs, reporting, and plotting.

## 1. Graph Generation (Discrete Points)
When updating or writing scripts that generate `.png` graphs (like sweeps in `Tests/`), the user requires exact numerical annotations for discrete points.
- ALWAYS add an arrow and the exact numerical text below or above every data point on the line.
- Offset the text dynamically so it does not alias or overlap with the main graph lines or axes.
- **Example implementation for annotations:**
  ```python
  arrow_props = dict(arrowstyle="->", color='purple', shrinkA=0, shrinkB=5)
  for i, val in enumerate(overall_accs):
      # Shift up if the point is too low, otherwise shift down
      y_offset = 25 if (val * 100) < 40 else -25
      ax.annotate(f"{val*100:.1f}%", (x_labels[i], val*100), 
                  textcoords="offset points", xytext=(0, y_offset), 
                  ha='center', fontsize=9, arrowprops=arrow_props)
  ```

## 2. Managing Markdown Reports (RTL & Paths)
- The main reports are stored in `Docs/` (e.g. `Docs/experiment_results.md`).
- Because reports are exported to PDF and HTML, they contain Hebrew text. You MUST maintain the embedded HTML block at the top of these documents to force RTL rendering.
- **Do not delete this block from the markdown files:**
  ```html
  <style>
  .markdown-body, body { direction: rtl !important; text-align: right !important; }
  .markdown-body ul, .markdown-body ol, .markdown-body p, .markdown-body h1, .markdown-body h2 { direction: rtl !important; text-align: right !important; }
  </style>
  ```
- **Image Paths:** Since markdown files are in `Docs/` but graphs are saved to `results/`, always use relative paths pointing up one directory when embedding images (e.g. `![Plot](../results/dim_sweep/snr_20/dim_sweep.png)`).

## 3. Scripts Organization
- DO NOT place new experiment scripts in the root directory.
- All new scripts that perform isolated research sweeps or evaluations must be placed in `Tests/`.
- If you add a new test, you must also append a new section explaining it to `Tests/README.md`.

## 4. Core Project Methodology & Architectural Wisdom
To prevent breaking the project's logic, any future modifications or agents MUST strictly adhere to these scientific principles we established:

- **Unsupervised Semantic Compression:** The Autoencoder (Encoder + Quantizer + Channel + Decoder) MUST NEVER see the classification labels during its training. Its sole objective is to reconstruct the original BERT embedding (using MSE loss). The classifier is just an evaluator to test how well the semantic meaning survived. **Do not add Cross-Entropy loss to the autoencoder's training loop!**
- **Channel-Awareness:** When testing performance across different noise levels (e.g. `snr_sweep.py`), a dedicated model MUST be trained specifically for each noise level. Do not evaluate a model on an SNR it wasn't trained for unless explicitly writing a "mismatch" test.
- **The "86% Baseline Illusion":** In the BERT semantic space, sentences cluster tightly together. The Cosine Similarity between any random sentence and the global mean is ~85.98%. Therefore, a reconstruction similarity of 88% in low dimensions is actually poor and leads to classification failure. Do not assume that >80% similarity means a successful reconstruction.
- **Caching & Efficiency:** When writing test sweeps, always implement a `--use_existing` flag (defaulting to 1). This prevents unnecessary retraining of models that already exist in the `results/` folder and allows quick iteration on plotting.
