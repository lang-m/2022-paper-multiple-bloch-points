# Pre-computed data

This directory containes pre-computed data files that require more computational
resources and time than typically available online, e.g. on Binder. All datasets
can be recomputed with [this notebook](../notebooks/create_data.ipynb).

## 80BP

- magnetisation data for 80 Bloch points encoding the word "Blochpoint" in ASCII
- length = 13.2 μm
- width = 100 nm
- `0_initial.hdf5`: magnetisation profile after energy minimisation of a
  suitable initial configuration
- `1_fieldpulse.hdf5`: magnetisation profile after applying a magnetic field
  pulse (B=0.25T in y-direction for t=0.5ns, time integration with LLG)
- `2_relaxation.hdf5`: magnetisation profile after turning of the external
  magnetic field and t=5ns of free relaxation (time integration with LLG)

## cellsize

- magnetisation data for one configuration containing 8 Bloch points
  (TT-HH-TT-TT-TT-TT-HH-TT)
- Three different cell sizes (cubic cells) to demonstrate the accuracy
  improvements with decreasing cell size
- `10.hdf5`: 1 nm cell size
- `25.hdf5`: 2.5 nm cell size
- `50.hdf5`: 5 nm cell size

## eight_bps.csv

- simulation results for 8 Bloch points in all possible combinations of HH and TT
- length = 1200 nm
- width = 100 nm

## figure3.csv

- simulation results for 1-3 Bloch points
- length in [100 nm, 600 nm] with 25 nm spacing
- width = 100 nm

## figure4.csv

- simulation results for 1-8 Bloch points of alternating type (HH-TT-HH-...)
- length in [100 nm, 1400 nm] with 20 nm spacing
- width in [100 nm, 200 nm] with 10 nm spacing
- fig4.csv is a subset of fig5.csv

## figure5.csv

- simulation results for 1-8 Bloch points of alternating type (HH-TT-HH-...)
- length in [100 nm, 1400 nm] with 20 nm spacing
- width in [100 nm, 200 nm] with 10 nm spacing
