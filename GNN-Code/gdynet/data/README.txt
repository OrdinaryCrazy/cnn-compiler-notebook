About the repository
====================

This repository contains trajectory files and preprocessed data files to reproduce the article "Graph Dynamical Networks for Unsupervised Learning of Atomic Scale Dynamics in Materials" by Tian Xie, Arthur France-Lanord, Yanming Wang, Yang Shao-Horn, and Jeffrey Grossman.

The files in this repository are in a format that can be used by the software package "gdynet" at https://github.com/txie-93/gdynet.


Description of files
====================

All the files are in .npz format which is a zipped archive of files named after the variables they contain. Each file can be considered as a dictionary-like data object which can be queried for its list of NumPy arrays. More details can be found at https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html#module-numpy.lib.format.

Trajectory files
----------------

The files `li2s-traj.npz`, `siau-traj.npz`, `peo-traj-a.npz`, `peo-traj-b.npz`, `peo-traj-c.npz`, `peo-traj-d.npz`, `peo-traj-e.npz` contain MD trajectories in the following format. They can be directly used by the preprocess script of the software "gdynet".

For a MD trajectory containing `N` atoms and `F` frames, the file has the following dictionary-like structure,

- `traj_coords`: `np.float` arrays with shape `(F, N, 3)`, stores the cartesian coordinates of each atom in each frame.
- `lattices`: `np.float` arrays with shape `(F, 3, 3)`, stores the lattice matrix of the simulation box in each frame. In the lattice matrix, each row represents a lattice vector.
- `atom_types`: `np.int` arrays with shape `(N,)`, stores the atomic number of each atom in the MD simulation.
- `target_index`: `np.int` arrays with shape `(n,)`, stores the indexes of the target atoms. (`n <= N`)

Preprocessed files
------------------

The files `li2s-traj-graph-train.npz`, `li2s-traj-graph-val.npz`, `li2s-traj-graph-test.npz` are preprocessed data files. They can be directly used by the software "gdynet" as the training, validation, and testing data.
