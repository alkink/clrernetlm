# LaneLM Debugging Journal

## 2024-05-22: Overfitting & Tokenization Refinement

### Context
- Initial attempts to overfit a single image failed with high loss (~2.5).
- Model was experiencing "Posterior Collapse", ignoring visual tokens.
- "One-to-Many" mapping problem identified: Single `BOS` token cannot map to 4 different lanes deterministically.

### Interventions & Experiments

#### 1. Tokenization Strategy: `relative_disjoint`
- Switched from `relative` (overlapping vocab) to `relative_disjoint` to prevent ambiguity between absolute coordinates and relative offsets.
- Settings: `nbins_x=200`, `max_abs_dx=32`. Vocab size approx 265.

#### 2. Visual Encoder: P5 Only
- **Observation:** Full FPN (P3-P5) provides ~5000 visual tokens, creating excessive noise for a small dataset.
- **Action:** Restricted encoder to **P5 Only** (lowest resolution, highest semantic level). Reduced tokens to ~200.
- **Result:** Loss dropped significantly from 0.81 to 0.56.

#### 3. Lane ID as Explicit BOS (Prompting)
- **Problem:** How does the model know WHICH lane to generate?
- **Solution:** Assigned unique BOS tokens for each lane index.
  - Lane 0 -> Token 296
  - Lane 1 -> Token 297
  - Lane 2 -> Token 298
  - Lane 3 -> Token 299
- **Rationale:** This aligns with "Prompt-based" generation. The BOS token acts as the query.

#### 4. Stability Fixes (Current)
- **Issue:** Loss dropped to 0.56 but then spiked to 0.94 (instability).
- **Fix:** Added `CosineAnnealingLR` and `Gradient Clipping` (norm=1.0) to `train_lanelm_overfit.py`.

### SUCCESS: Strict Overfit Achieved (2024-11-20)
- **Configuration:** P5 Only + Explicit BOS + Scheduler (Cosine) + Clip Grad (0.5) + LR (3e-4 -> 1e-6).
- **Result:** Loss dropped to **0.26** and remained stable.
- **Conclusion:** The architecture can perfectly memorize 4 distinct lanes when visual noise is reduced and intent (Lane ID) is explicit.

### Next Steps
- **Step 1:** Verify visual output (Ep 1000 images).
- **Step 2:** Scale up to "Mini-Batch Overfit" (e.g., 8-16 images) to test generalization capacity.
- **Step 3:** Re-integrate into full training pipeline with these settings.
