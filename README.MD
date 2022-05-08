# Data Intelligence Challenge solution

## Setup
We used pipenv for setting up our virtual environment. It fully specifies dependencies and subdependencies 
in the Pipfile and Pipfile.lock.

For documentation see https://pipenv.pypa.io/en/latest/ 

### Steps:
1. Activate or install a Python 3.8 environment and install pipenv in it.
   - For instance using conda: `conda create -n py38 python=3.8.12 pipenv -c conda-forge`
   - Don't forget to activate it or add it to your path variable.
2. Replicate the project environment by running `pipenv install` in the project folder.
3. Activate the project environment with `pipenv shell` or configure your IDE to use the new pipenv environment (with the project folder name) as the python environment.

Note, that the C++ build tools on your system are required for using this project.

## Notes

Cel material | Label
-------------|-------
Obstacle (gray)    | -2
Wall (red)         | -1
Clean (green)      |  0
Dirty (white)      |  1
Goal (orange)      |  2
Death (red cross)  |  3