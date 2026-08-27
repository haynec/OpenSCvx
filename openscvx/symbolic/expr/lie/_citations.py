"""Shared BibTeX entry for the jaxlie-backed Lie-group nodes.

The SO(3)/SE(3) exponential and logarithm maps in this package lower through
:mod:`jaxlie`; its authors ask users to cite the paper below. The entry lives
here once so ``so3.py`` and ``se3.py`` can both return it from ``citation()``.
"""

JAXLIE_CITATION = r"""@inproceedings{yi2021iros,
  title={Differentiable Factor Graph Optimization for Learning Smoothers},
  author={Brent Yi and Michelle Lee and Alina Kloss and Roberto Mart\'in-Mart\'in and
    Jeannette Bohg},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021}
}"""
