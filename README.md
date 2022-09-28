<!--
 * @Date: 2022-15-Tu 20:15:53
 * @LastEditors: SHAO Nuoya
 * @LastEditTime: 2022-09-28 16:10:08
 * @FilePath: \JLT_Projet\readme.md
-->
# JLT model, construction and calibration

## Requirements
- Anaconda \
    `Make sure that the path has been added whle installing Anaconda, though the programme said it's not recommanded`
- Vs code (recommanded)

## Configurations
- add project path to PYTHONPATH (optional)
- run commands as below in cmd:
    * `cd [your project path]`
    * `pip install -r requirements.txt`
- in JLT/main.py, replace **sys.path.append("C:\\Users\\SHAO\\Desktop\\JLT_Projet")** by
  **sys.path.append("[your project path]")**
- if you want to recreate the run.exe, run create_exe.py. The .exe will appear in document
  dist, you can change his name and his path as your wish

## Check documentation
- open cmd and run command
    * `cd [your project path]`
    * `mkdocs serve`
- copy the link appeared in cmd and open it with navigator

## Run progrmme
- double click run.exe
- select mode and choose appropriate parameters \
- Parameters explanation
    * **Mode** :
        - CALI : calibrate JLT model
        - SHOW : show the calibration result with parameter in [Result path]
        - ESG  : create ESG document (.csv)
    * **Algo** : \
        different algorithms to calibrate JLT model (algos for scipy.optimizer.minimize)
    * **lbound, ubound, n** : \
        lower bound and upper bound for parameters, n is the number of initial values for each parameter. For example, for "alpha lbound=0, ubound=5, n=3", alpha will be whithin [0, 5] and we will use 1, 2, 3 as initial values of alpha
    * **Core numbers** : \
        number of core that will be used for parallel operation (maximum 8 for Allianz's computer)
    * **Bond type** : \
        CORP or GOUV
    * **Ratings, years, values** : \
        parameters for setting weights 
        - Ratings : the ratings whose weights will be reset
        - Years : specific maturites where we want to reset weights
        - Values : the value of weight that we want to set
    * **Para path** : \
        The path 
        - where current calibration result will be stored
        - with which we generate the ESG
        - with which we will show the calibration result
