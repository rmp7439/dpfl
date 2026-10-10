# Privacy Accounting Audit

## Date: 2026-10-10
## Auditor: Independent internal review (Agent)

### Summary of Findings
The RDP privacy accounting implementation in `scripts/run_federated.py` correctly calculates empirical global privacy loss, but lacks formal external peer review. The methodology adheres to standard parallel composition under disjoint client datasets.

### Verification Points

1. **Per-step RDP cost and assumptions**: The accountant uses the empirical sample rate $q = \text{batch\_size} / \text{local\_dataset\_size}$. The per-step cost correctly uses `compute_rdp` from the opacus accountant package.
2. **Composition**: The RDP state per client is composed sequentially over the number of actual DP steps they perform each round.
3. **Conversion**: RDP is converted to $(\varepsilon, \delta)$-DP at the end of each round using $\delta=10^{-5}$ via Opacus's `get_privacy_spent`.
4. **Optimal Rényi Order**: Selected automatically per client by `get_privacy_spent(orders=alphas, rdp=...)`.
5. **Heterogeneity Impact**: Local dataset sizes heavily influence the sample rate $q$. Because data is Dirichlet-partitioned ($\alpha=0.1$), client dataset sizes vary significantly, thus privacy costs per step vary.
6. **Parallel Composition**: The maximum $\varepsilon$ among all clients is reported as the global privacy guarantee. This parallel composition is mathematically sound *only* because the Dirichlet partition creates disjoint datasets; thus, a single sample appears in exactly one client's dataset.
7. **Artifact Source**: Epsilons in the report match those computed dynamically and saved in `summary.json`. 

### Outstanding Actions
- **Peer Review Status**: Formal external human peer review of the RDP code path and parallel composition justification remains **OUTSTANDING**. Currently, it has only been internally verified.

### Conclusion
The code accurately calculates epsilon under its assumptions. No mathematical discrepancies were found in the conversion or composition logic, but the lack of formal external peer review must be noted in all publications.
