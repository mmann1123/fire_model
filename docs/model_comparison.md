# Fire Model: Experiment Comparison

## Run History

| Run | Date | Model | Description | AUC-A | AUC-B | Delta | Notes |
|-----|------|-------|-------------|-------|-------|-------|-------|
| v1-baseline | 2026-04-01 | LogReg | All features, 300-acre min, balanced weights | 0.9226 | 0.9207 | +0.0019 | Initial baseline. Emulator operationally viable. |

## Key Findings

### v1-baseline
- **Overall:** Delta AUC +0.0019 — emulator degrades fire skill by <0.2%
- **Fire season (Jun-Nov):** Track A 0.8583, Track B 0.8573 (Delta +0.0011)
- **WY2021 (Dixie Fire):** Emulator AUC 0.9228 vs BCMv8 0.8989 — emulator captures megadrought signal better (-0.0238 delta in emulator's favor)
- **WY2022:** Emulator outperforms (0.8551 vs 0.8380, -0.0171 delta)
- **WY2024:** Emulator outperforms (0.9267 vs 0.9214, -0.0053 delta)
- **Top features by permutation importance:** tmax (+0.393), vpd (+0.057), dist_airbase (+0.043), sws (+0.043), tsf_years (+0.032), tmin (+0.032)
- **Top features by |coefficient|:** tmax (+4.10), vpd (-1.70), tmin (-1.31), sws (-1.21), dist_airbase (-1.16), kbdi (-0.76)
- **Conclusion:** Emulator is operationally viable for fire prediction
