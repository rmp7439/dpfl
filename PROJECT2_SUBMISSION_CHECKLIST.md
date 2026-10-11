# Project 2 Submission Checklist

[PASS] Dirichlet α=0.1 non-IID split confirmed
[PASS] per-client label histogram exists in report
[PASS] Flower federation runs end-to-end
[PASS] DP-SGD uses Opacus PrivacyEngine
[PASS] per-sample clipping verified
[PASS] Gaussian noise verified
[PASS] RDP accounting documented
[PASS] full chain:
    per-step RDP
    → T-step composition
    → (ε,δ)-DP conversion
[PASS] every privacy run has:
    σ
    C
    δ
    Rényi order
    ε
[PASS] Stage 5 σ/C ablation complete
[PASS] accuracy-vs-ε figure exists
[PASS] convergence graph exists
[PASS] centralized baseline comparison exists
[PASS] α=0.1 vs α=10 comparison exists
[PASS] σ=2.0 comparison explicitly discussed
[PASS] Stage 6 robustness status explicitly documented (Audited & verified: 4 authoritative 15-round full CIFAR-10 reference runs + 12-run 3-round multi-seed empirical analysis across seeds 42, 43, 44 documented in results/stage6/STAGE6_SUMMARY.md and technical_report.tex)
[PASS] non-private 15-round FedAvg control status documented
[PASS] baseline 74.87% caveat documented
[PASS] LaTeX report compiles
[PASS] figures are regenerated
[PASS] publication-quality figures ≥300 dpi where applicable
[PASS] README complete
[PASS] requirements/environment reproducibility documented
[PASS] RNG seeds documented
[PASS] no stale numerical claims remain
[PASS] no unsupported causal claims remain
[PASS] tests run and results documented
