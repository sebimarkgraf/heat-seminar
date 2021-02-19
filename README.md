# Using HeAT for High-Performance Clustering of Remote-Sensing Data

This work was created during the "Big Data Tools" Seminar 2020 at the Karlsruhe Institute for Technology.
My task was to use [HeAT](https://github.com/helmholtz-analytics/heat) to test unsupervised methods on remote-sensing data.

The focus was mainly on the scaling properties of HeAT. The clustering itself was not really successful, which is due to the used pixelwise distances for image data.
For more information about the work here, please refer to the included paper.


## Installing
Create a virtual environment and install all the dependencies, e.g.
``` bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the data file for SO2SAT in a directory.
Configure the location in  `code/config-defaults.yaml` or through any other method supported by WanDB.

If you want to use the WanDB logging, please follow the [getting started guide](https://wandb.ai/site/getting-started) from WanDB


## Citing

If you want to cite this work, please use the followin BibTex entry.

``` bibtex
@article{markgraf_using_2020,
	title = {Using {HeAT} for {High}-{Perfomance} {Clustering} of {Remote}-{Sensing} {Data}},
	language = {en},
	author = {Markgraf, Sebastian and Debus, Charlotte},
	month = mar,
	year = {2020},
	pages = {8}
}
```
