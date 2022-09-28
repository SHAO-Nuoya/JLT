# JLT model, construction and calibration

## Prerequisites
- add project path to PYTHONPATH
- run commands as below:
    * `cd "project path"`
    * `pip install -r requirements.txt`

## Check documentation
- open cmd and run commands as below
    * `cd "project path"`
    * `mkdocs serve`
- open the link in cmd with navigator

## Generate .exe file
`pyinstaller -F -w tk.py`

## Run progrmme
- click run.exe
- select mode and choose appropriate parameters

## Project layout
    │   .gitignore
    │   .secrets.toml
    │   config.py
    │   readme.md
    │   requirements.txt
    │   run.exe
    │   settings.toml
    │   tk.py
    │   tk.spec
    │   __init__.py
    │
    │
    │
    ├───JLT
    │   │   calibrate.py
    │   │   db.py
    │   │   diffuse.py
    │   │   input.py
    │   │   main.py
    │   │   mkdocs.yml
    │   │   test.py
    │   │   utility.py
    │   │   __init__.py
    │   │
    │   ├───data
    │   │       CS_31122021.xlsx
    │   │       TransitionProbabilityMatrix.csv
    │   │
    │   ├───docs
    │   │       calibrate.md
    │   │       index.md
    │   │       input.md
    │   │       JLT_model.md
    │   │       main.md
    │   │       random.md
    │   │       utility.md
    │   │
    │   └───Result
    │       ├───calibration
    │       │   ├───CONSTR
    │       │   ├───Powell
    │       │   └───SLSQP
    │       │
    │       ├───ESG
    │       ├───other
    │       └───random_test
    │    
    │
    └───References
            A Markov Model for the Term Structure of Credit Risk Spreads.pdf
            A Stochastic Model for Credit Spreads under a Risk-Neutral.pdf
            Finding Generators for Markov Chains.pdf
            Generate CIR Process.pdf
            Good Parameters and Implementations for Combined.pdf
            Guillaume_Bellier_Memoire_Actuaire.pdf
            MNA Implémentation d’un modèle de crédit.pdf
            Moments of CIR.pdf
            On Cox Processes and Credit Risky Securities.pdf




# About MkDocs
For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands
* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.