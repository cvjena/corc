# corc - Computation of Radial Curves

This project contains the source code to compute the `Volumetric Facial Palsy Grading` published at the [MIE-2022](https://mie2022.org/). You can find the accompanying paper [here](todo).
If you use any part of the code for your own research or development, please use the provided citation at the end of the document.

## Overview

The code features several processing steps which can be used independently from each other:

*  Loading of different facial landmark formats:
    * Basic 68 system (dlib)
    * 3DFE
    * 3DFE-MVLM
    * BP4D
    * Custom landmarking system by implementing an inheriting class of `corc.landmarks.Landmarks`.
* Loading of different point-cloud/mesh formats:
    * `.ply`
    * `.obj`
    * `.wrl`
* Computation of `Radial Curves` using the 3D landmarks and the point clouds
* Computation of the volumetric score using the `z` point-wise mean difference

All calculations are done in `numpy`, `scipy`, and `igraph` on the CPU.
We utilize multiprocessing during the extraction and processing of the `Radial Curves` to achieve almost real-time results.

## Installation

**Note:** The system has been developed and extensively been tested only on Linux based systems. For `Windows` and `MacOS` based system please open an issue of problems occur.
We assume you have a proper python installation on your system and using either a [virtual environment](https://docs.python.org/3/tutorial/venv.html) or [anaconda](https://www.anaconda.com/).
One can create a `conda` environment using:
```sh
conda create -n corc-test python=3.9 pip -y
```


To install the `corc` please copy the repository into a directory of your choise.

```sh
git clone git@github.com:cvjena/corc.git
```

Inside the folder please run
```sh
pip install .
```

Now the the package should be available in your python installation via

```python
import corc
```

The following sub-modules are available for usage inside the package

```python
from corc import core, grading, landmarks, utils
```

## Basic Usage

This is a minimal example on how to use the `corc` library to compute the `Radial Curves` from a given 3D landmark and a point-cloud file.
For extended usage please refer to the doc-strings of the functions.
The algorithm usage can be fine-tuned using the defined keyword arguments.

```python
import pathlib

import numpy as np

from corc import core, grading, landmarks, utils

# we assume you use the basic 68 landmark system in this example
lm_file = pathlib.Path("path/to/your/landmarks_file.csv")
pc_file = pathlib.Path("path/to/your/pointcloud_file.csv")

# the loaded files will be in the form of a numpy array.
lm: np.ndarray = utils.load_landmarks(lm_file)
pc: np.ndarray = utils.load_pointcloud(pc_file)

# the landmarks have to forwarded to the correct Landmark class
# this class handles the correct access of the specific landmark features
# like eyes, nose-tip, mouth
radial_curves: np.ndarray = core.compute_corc(
    point_cloud=pc,
    landmarks=landmarks.Landmarks68(lm),
    delta=0.015,
    n_curves=128,
    n_points=128,
)

# the computed radial curves are also in the form of the numpy array
# in this example they have the form of 128x128x3
grad: float = grading.volume_grading(radial_curves, n_curves=128, n_points=128)
```

## Citation

If you use this work please use the following bibtext entry:

```tex
coming soon
```
