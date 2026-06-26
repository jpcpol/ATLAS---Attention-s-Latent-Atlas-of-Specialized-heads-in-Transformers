# License

Copyright © 2026 Juan Pablo Chancay (jpcpol@gmail.com).

This project uses a **dual license** that distinguishes research/written material from
executable code. Which license applies depends on the part of the repository:

| Part of the repository | License |
|---|---|
| **Documents & theory** — everything in `docs/` and `theory/`, plus `README.md`, `CLAUDE.md`, and the figures in `docs/figures/` | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** |
| **Source code** — everything in `src/`, `experiments/` scripts, and any other executable code | **GNU Affero General Public License, version 3.0 (AGPL-3.0)** |

If a single file could plausibly fall under both (e.g. a Markdown file containing a code
snippet), the **prose** is under CC BY-NC 4.0 and the **code snippet** is under AGPL-3.0.

---

## 1. Documents and theory — CC BY-NC 4.0

The written content (papers, theory notes, research questions, figures, this README) is licensed
under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

You are free to:

- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

Under the following terms:

- **Attribution** — You must give appropriate credit to *Juan Pablo Chancay*, provide a link to
  this license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.

Full legal text: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>
Human-readable summary: <https://creativecommons.org/licenses/by-nc/4.0/>

> **How to cite.** If you use the results, figures, or framing of this work, please cite the
> preprint *"A Scale-Invariant Atlas of Head-Specific Manifolds in Transformer Residual
> Attention"* (Chancay, 2026) — see [docs/paper_draft.md](docs/paper_draft.md).

---

## 2. Source code — AGPL-3.0

All source code in this repository is licensed under the **GNU Affero General Public License,
version 3.0**.

```
This program is free software: you can redistribute it and/or modify it under the terms of
the GNU Affero General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this
program. If not, see <https://www.gnu.org/licenses/>.
```

The AGPL-3.0 in particular requires that if you run a modified version of this code to provide
a service over a network, you must make the corresponding source of your modified version
available to the users of that service.

Full legal text: <https://www.gnu.org/licenses/agpl-3.0.txt>

---

## Why this combination

- **CC BY-NC 4.0** keeps the scientific narrative open for academic reuse, replication, and
  teaching while reserving commercial exploitation of the writing.
- **AGPL-3.0** keeps the measurement and analysis code open and copyleft, including over a
  network — appropriate for research tooling whose value is in being inspectable and reproducible.

This matches the licensing of the related CAL line, as noted in [CLAUDE.md](CLAUDE.md).

---

## No warranty

The material is provided "as is", without warranty of any kind, express or implied. This is a
research artifact: the founding quantization hypothesis (NQP) was empirically refuted, and the
surviving results are reported with their stated scope and limitations. Nothing here constitutes
a production-ready method.
