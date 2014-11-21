step-35a
========
Overview
--------
This project is a modified copy of step-35 in deal.II. The important changes are
* The code can now run in 3D (some of the `DoFHandler` code needed to be modifed)
* The code now outputs binary HDF5 files

Files
-----
This project comes with a few sample meshes:
1. `cylinder3d.msh` : a 3D flow past a cylinder problem.
2. `nsbench2.inp`   : a 2D flow past a (square) cylinder problem.

Authors
-------
* Abner Salgado, Texas A&M
* David Wells, Virginia Tech
* The deal.II team