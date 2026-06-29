# References consulted (decision trail)

Papers consulted during the AICR cycles that **informed decisions** — kept for traceability and
honest attribution (which paper grounded which choice). Status column: `in paper` = already cited in
`docs/paper_draft.md`; `integrate` = should be added; `context` = informed a decision but may not need
a citation (judgment call when integrating).

| ref | paper | used for | status |
|---|---|---|---|
| [18] | Valeriani et al., *The geometry of hidden representations of large transformer models* (arXiv:2302.00294) | ID profile phases (expansion→compression→ascent); peak grows with size / plateau constant; rel-depth locates phases. Grounded: rel-0.9→0.5 measurement switch, G0b early-peak check, "scale not the lever". | **in paper** [18] |
| — | *Quality over Quantity in Attention Layers* (ICLR 2025) | The d_head-vs-n_head question directly; strong venue. Informs batch-2 factorial value + §3.1c positioning. NOT YET READ in full. | **integrate** (read first) |
| — | *A Capacity-Based Rationale for Multi-Head Attention* (arXiv:2509.22840) | Opposite pole of the debate: "many small heads reduce interference, more separable relations in overlapping subspaces." The d_head/n_head contradiction the factorial can arbitrate. | **integrate** |
| — | *The Spike, the Sparse and the Sink* (arXiv:2603.05498) | d_head dominant for **attention sinks** (8→128 monotonic). RELATED but DISTINCT phenomenon (sinks ≠ inter-head residual overlap). Cite WITH the distinction — NOT as confirmation of our O_h result (anti-NQP: nearly cited it wrongly, source-tracing caught it). | **integrate** (with caveat) |
| — | *Analyzing and Controlling Inter-Head Diversity in MHA* (MDPI Appl.Sci. 11/4/1548) | Inter-head diversity via SVCCA/CKA — sister metrics to our O_h; "optimal inter-head diversity exists". Contextualizes O_h as a recognized object. | **integrate** |
| — | Head-redundancy literature (BERT 70–90% heads removable; power-law head importance) | Why high O_h = redundant heads is plausible. Background for §3.1b/§5. | context |
| — | *Attention Layers Add Into Low-Dimensional Residual Subspaces* (arXiv:2508.16929) | Attention-output low-rankness is partly architectural (W^O, dim(⋃ span) ≤ d_head·n_head); EXPLICITLY does not test init-vs-trained. The open gap H-TEMP targets. | integrate (H-TEMP line) |
| — | Ansuini et al., *Intrinsic dimension of data representations in DNNs* (arXiv:1905.12784) | ID rises-then-compresses during training — confirms OBS-A's plateau-d_int rise→decay as known. | integrate (H-TEMP line) |
| — | Functional head-emergence lit (induction/retrieval staged emergence; arXiv:2502.06902, 2411.12118, 2404.07129) | Functional heads emerge gradually ≠ our subspace geometry — resolves the apparent OBS-A contradiction. | context (H-TEMP) |
| — | Post-treatment mediator bias (causal-inference lit; arXiv:2107.11014, PMC7456832) | Confirmed a d_head→d_int→O_h mediation analysis would be biased → kept it OUT of the paper. | context (methods) |
| — | RG for deep networks (arXiv:2510.25553) | RG universality framing — used to JUSTIFY softening "universality class" → "fixed-point-like". | context (framing) |

**Note on numbering:** when integrating, assign [24]+ in `paper_draft.md` references and add the
"we differ by …" sentence for each in Related Work, the same discipline used for [17]–[23].
**Verify metadata** (authors, year, venue) for each before final submission — several are arXiv
preprints found via search; confirm via a reference manager.
