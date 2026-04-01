# Fire Model: Experiment Comparison

## Run History

| Run | Date | Model | Description | AUC-A | AUC-B | Delta | Notes |
|-----|------|-------|-------------|-------|-------|-------|-------|
| v1-baseline | 2026-04-01 | LogReg | All features, 300-acre min, balanced weights | 0.9139 | 0.9131 | +0.0008 | Initial baseline. Emulator operationally viable. |

## Key Findings

### v1-baseline
- **Overall:** Delta AUC +0.0008 — emulator degrades fire skill by <0.1%
- **Fire season (Jun-Nov):** Track B outperforms Track A by -0.0022 AUC
- **WY2021 (Dixie Fire):** Emulator AUC 0.908 vs BCMv8 0.878 — emulator captures megadrought signal better
- **Top features:** tmax, vpd, tmin, sws, kbdi (climate dominates; CWD anomalies rank 18-21)
- **Conclusion:** Emulator is operationally viable for fire prediction
