# Reproducible Deep Learning
## Extra: Python-Poetry
Author: Timur Obukhov
There are several tools in Python space for dependency and workspace management. The table below shows a comparison between the most prominent tools: 

```bash
┌──────────────────┬────────────────────┬──────────────────────────┬───────────────────────┬───────────────────────┐
│                  │   Python Version   │  Dependency Management   │  Virtual Environment  │   Env Reproducibility │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ pyenv            │         YES        │            NO            │           NO          │       NO              │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ venv + pip       │         NO         │            YES           │           YES         │       NO              │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ venv + pip-tools │         NO         │            YES           │           YES         │       YES             │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ Poetry           │         NO         │            YES           │           YES         │       YES             │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ Pipenv           │         NO         │            YES           │           YES         │       YES             │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ Docker           │         NO         │            NO            │           NO          │       YES             │
├──────────────────┼────────────────────┼──────────────────────────┼───────────────────────┼───────────────────────┤
│ Conda            │         YES        │            YES           │           YES         │       NO              │
└──────────────────┴────────────────────┴──────────────────────────┴───────────────────────┴───────────────────────┘
```
The choice for dependency management and reproducibility is Poetry. 
Out of the options presented above, Poetry provides an exhaustive dependency resolver. In addition, poetry configures a virtual environment for the project.  Poetry commands are intuitive and easy to use. 

## Installation

```Python-Poetry``` for ```Windows``` installation instruction: 

```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -
```

> ⚠️ The previous ```get-poetry.py``` installer is deprecated.

The installer installs the poetry tool to ```Poetry's``` bin directory. For ```Windows```, it is in:

```bash
%APPDATA%\Python\Scripts
```

If this directory is not on your PATH, you will need to add it manually to invoke Poetry with simply ```poetry```.

It is also possible to install Poetry from a ```git``` repository by using ```--git``` option: 

```bash
python install-poetry.py --git https://github.com/python-poetry/poetry.git@master
```

For Linux/Mac please see installation instructions here:[Potery installation](https://python-poetry.org/docs/)

## Use of Poetry
### Creating and managing a new project: 

When ```Poetry``` is installed, we can create a new project: 

```bash
$ poetry new reprodl2021_extra_poetry
$ cd reprodl2021_extra_poetry
```

This creates the following folder structure: 

```bash
reprodDL2021_extra_poetry
├── LICENSE
├── README.md
├── pyproject.toml
├── poetry.lock
├── reprodl_overview.png
├── reproddl2021_extra_poetry
│   ├── Initial Notebook.ipynb
│   └── __init__.py
└── tests
    ├── Initial Notebook.ipynb
    └── __init__.py
```

Dependencies are declared  are stored in the ```pyproject.toml``` file:

```bash
[tool.poetry]
name = "reprodDL2021_extra_poetry"
version = "0.1.0"
description = ""
authors = ["Timur Obukhov <obukhov@diag.uniroma1.it>"]

[tool.poetry.dependencies]
python = "^3.8"
matplotlib = "3.4.1"
torch = "1.8.1"
numpy = "1.20.2"
hydra-core = "1.0.6"
pandas = "1.2.3"
pytorch-lightning = "1.3.0"
tqdm = "4.60.0"
torchaudio = "0.8.1"
omegaconf = "2.0.6"
librosa = "0.8.0"
seaborn = "0.11.1"

[tool.poetry.dev-dependencies]
pytest = "^5.2"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```
### Poetry commands: 

Poetry provides various commands to build and add dependencies, track and publish packages, and other manipulation with packages.  
Avaibale commands for command-line can be found here: [Commands for Potery](https://python-poetry.org/docs/cli/).

For example, to add dependencies, run the following command:  

```bash
$ poetry add flask 
```
This command will download and install ```Flask``` from ```PyPI``` in the virtual environment managed by Poetry and add this to ```poetry.lock``` file and adds it to top-level dependency to ```pyproject.toml```

```bash
[tool.poetry.dependencies]
python = "^3.8"
flask = "^1.1.2"
```

### Virtual enviroments:

Poetry makes project environment isolation one of its core features.
To get basic information about the currently activated virtual environment use  ```env info``` command:

```bash
poetry env info 
```
This provides the following output: 

```bash
Virtualenv
Python:         3.8
Implementation: CPython
Path:           /path/to/poetry/cache/virtualenvs/reprodDL2021_extra_poetry07e py3.8
Valid:          True

System
Platform: Win32
OS:       nt
Python:   /path/to/main/python
```

For more information on virtual environment, please follow this link: [Virtualenv](https://python-poetry.org/docs/managing-environments/)

### Conclusion: 

This exercise looked at the use of tools for dependency management. For the experiment, Poetry was selected. 
We experimented with installing and switching between different versions and also managing dependencies. 
