---
title: Citation
description: >-
  How to cite OpenSCvx and the successive-convexification methods it implements,
  including Hayner et al., IEEE RA-L 2025, in BibTeX.
---

# Citation

If you use OpenSCvx in your research, please cite Hayner et al., *Continuous-Time
Line-of-Sight Constrained Trajectory Planning for 6-Degree of Freedom Systems*,
IEEE Robotics and Automation Letters (RA-L), 2025
([DOI: 10.1109/LRA.2025.3545299](https://doi.org/10.1109/LRA.2025.3545299)),
along with the methodological and solver references below.

## Primary Citation

Hayner et al., IEEE RA-L 2025 — DOI:
[10.1109/LRA.2025.3545299](https://doi.org/10.1109/LRA.2025.3545299).

```bibtex
@ARTICLE{hayner2025los,
        author={Hayner, Christopher R. and Carson III, John M. and Açıkmeşe, Behçet and Leung, Karen},
        journal={IEEE Robotics and Automation Letters}, 
        title={Continuous-Time Line-of-Sight Constrained Trajectory Planning for 6-Degree of Freedom Systems}, 
        year={2025},
        volume={},
        number={},
        pages={1-8},
        keywords={Robot sensing systems;Vectors;Vehicle dynamics;Line-of-sight propagation;Trajectory planning;Trajectory optimization;Quadrotors;Nonlinear dynamical systems;Heuristic algorithms;Convergence;Constrained Motion Planning;Optimization and Optimal Control;Aerial Systems: Perception and Autonomy},
        doi={10.1109/LRA.2025.3545299}}
```

## Methodological Foundation

Elango et al., *Successive Convexification for Trajectory Optimization with
Continuous-Time Constraint Satisfaction*, 2024 — preprint:
[arXiv:2404.16826](https://arxiv.org/abs/2404.16826).

```bibtex
@misc{elango2024ctscvx,
      title={Successive Convexification for Trajectory Optimization with Continuous-Time Constraint Satisfaction}, 
      author={Purnanand Elango and Dayou Luo and Abhinav G. Kamath and Samet Uzun and Taewan Kim and Behçet Açıkmeşe},
      year={2024},
      eprint={2404.16826},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2404.16826}, 
}
```

## Solver Technology

Chari and Açıkmeşe, *QOCO: A Quadratic Objective Conic Optimizer with Custom
Solver Generation*, 2025 — preprint:
[arXiv:2503.12658](https://arxiv.org/abs/2503.12658).

```bibtex
@misc{chari2025qoco,
  title = {QOCO: A Quadratic Objective Conic Optimizer with Custom Solver Generation},
  author = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year = {2025},
  eprint = {2503.12658},
  archiveprefix = {arXiv},
  primaryclass = {math.OC},
}
```

## Citing the software

To cite the OpenSCvx software itself, use the
[`CITATION.cff`](https://github.com/OpenSCvx/OpenSCvx/blob/main/CITATION.cff) file
at the repository root — GitHub's "Cite this repository" widget reads it directly.
It points to the RA-L paper above as the preferred citation. A dedicated software
DOI (Zenodo) is planned for the 1.0 release.

## Acknowledgments

This work was supported by a NASA Space Technology Graduate Research Opportunity and the Office of Naval Research under grant N00014-17-1-2433. The authors would like to acknowledge Natalia Pavlasek, Samuel Buckner, Abhi Kamath, Govind Chari, and Purnanand Elango as well as the other Autonomous Controls Laboratory members, for their many helpful discussions and support throughout this work. 