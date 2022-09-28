'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Wednesday, July 6th 2022, 1:06:04 pm
   Last Update:    Wednesday, July 6th 2022, 5:53:43 pm
   File:           create_exe.py
   Copyright (c) 2022 Allianz
'''
import PyInstaller.__main__
PyInstaller.__main__.run(["tk.py", "--onefile", "--windowed"])