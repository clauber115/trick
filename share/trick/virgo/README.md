# VIRGO

VIRGO: **V**ersatile **I**maging and **R**endering for **G**alactic **O**perations. A Practical, Analytical, and Hardworking 3D Visualization tool for Trick-produced data leveraging python-VTK

## Module Dependencies

This module requires python3.11 or later and the `pip` packages listed below. To pull down these packages to a standard python3 virtual environment, create `requirements.txt` with the following content:

```
numpy
PyYAML
vtk
pandas # for trickpy
```
Then, create a python3.11 virtual env anywhere you wish using the packages in `requirements.txt`:
```bash
# cd to the path you want to create .venv in, then run...
python3.11 -m venv .venv && source .venv/bin/activate && pip3 install --upgrade pip && pip3 install -r requirements.txt
```
Once the `.venv` is created, you can source the environment in any shell before running scripts using VIRGO:

```bash
source .venv/bin/activate
```

