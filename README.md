# FOV Security Research Project

This project is for the FOV security research task. This is a fork from the [`avdev-sandbox`][sandbox] that has all the `AVstack` dependencies set up.

The FOV-specific analysis tools are under the `fov_analysis` folder.

## Contributing

To avoid adding large jupyter notebooks with images etc., please add the pre-commit hook to your git pipeline via `git config core.hooksPath hooks` in the terminal. This will wipe the notebooks of any cell output every time you commit. Ensure when you make a pull request that the notebooks have wiped the output.

## Installation

TLDR:
```
git clone --recurse-submodules https://github.com/cpsl-research/fov-security
cd fov-security
poetry install
git config core.hooksPath hooks
./initialize.sh /data/shared /data/shared/models
poetry run jupyter notebook
```

### Requirements
This currently only works on a Linux distribution (tested on Ubuntu 20.04 and 22.04). It also only works with Python 3.10. [Poetry][poetry] must be installed on your system to handle the dependencies. Python 3.10 must be installed on your system.

#### Troubleshooting

- If you install poetry but your systems says it is not found, you may need to add the poetry path to your path. On linux, this would be: `export PATH="$HOME/.local/bin:$PATH"`. I recommend adding this to your `.bashrc` or `.zshrc` file.
- Through an ssh connection, poetry may have keyring issues. If this is true, you can run the following: `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`


### Installation
```
git clone --recurse-submodules https://github.com/cpsl-research/fov-security
cd fov-security
poetry install  # to install the dependencies
```

### Execute Basic Tests
Try the following and see if it works.
```
cd examples/hello_world
poetry run python hello_import.py
```
This will validate whether we can import `avstack` and `avapi`.

### Download Models and Datasets
To get fancy with it, you'll need perception models and datasets. To install those, run
```
./initialize.sh  # to download models and datasets
```
The initialization process may take a while -- it downloads perception models and AV datasets from our hosted data buckets. If you have a preferred place to store data and perception models, you can pass that as an argument by running:
```
./initialize.sh /path/to/save/data /path/to/save/models
```

### Execute Full Tests
Once this is finished, let's try out some more interesting tests such as
```
cd examples/hello_world
poetry run python hello_api.py
```
which will check if we can find the datasets we downloaded.

And
```
cd examples/hello_world
poetry run python hello_perception.py
```
which will check if we can properly set up perception models using `MMDetection`.

### Fire Up Jupyter Notebooks
Now that you have the basic tests running, fire up the jupyter notebooks to get in to some more involved experimentation. You can once again do this through poetry by running
```
poetry run jupyter notebook
```
Then go into `examples/notebooks` and start playing around with them.


### Reporting Bugs

I welcome feedback from the community on bugs with this and other repos. Please put up an issue when you find a problem or need more clarification on how to start.

# LICENSE

Copyright 2023 Spencer Hallyburton

AVstack specific code is distributed under the MIT License.



[rtd-page]: https://avstack.readthedocs.io/en/latest/
[core]: https://github.com/avstack-lab/lib-avstack-core
[api]: https://github.com/avstack-lab/lib-avstack-api
[sandbox]: https://github.com/avstack-lab/avdev-sandbox
[avstack-preprint]: https://arxiv.org/pdf/2212.13857.pdf
[poetry]: https://github.com/python-poetry/poetry
[mmdet-modelzoo]: https://mmdetection.readthedocs.io/en/stable/model_zoo.html
[mmdet3d-modelzoo]: https://mmdetection3d.readthedocs.io/en/stable/model_zoo.html
[contributing]: https://github.com/avstack-lab/lib-avstack-core/blob/main/CONTRIBUTING.md
[license]: https://github.com/avstack-lab/lib-avstack-core/blob/main/LICENSE.md

