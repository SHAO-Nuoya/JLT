'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Wednesday, June 29th 2022, 2:19:45 pm
   Last Update:    Monday, July 4th 2022, 10:17:31 am
   File:           tk.py
   Copyright (c) 2022 Allianz
'''
import tkinter as tk
from tkinter.ttk import Style
from toml_config.core import Config
import ast
from tkinter import INSERT
import os
    
class Windows:
    def __init__(self) -> None:
        window = tk.Tk()
        window.title("JLT project")
        window.geometry('700x450')
        
        style = Style()
        style.theme_create( "MyStyle", parent="alt", 
                           settings={"TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
                                     "TNotebook.Tab": {"configure": {"padding": [10, 5] },}
                                     }
                           )
        style.theme_use("MyStyle")

        notebook = tk.ttk.Notebook(window)
        self.frameOne = tk.Frame()
        self.frameTwo = tk.Frame()
        notebook.add(self.frameOne, text="JLT")
        notebook.add(self.frameTwo, text="Vasicek")
        notebook.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.window = window
        self.notebook = notebook
        
        self.set_function()
        self.set_weight()
        self.set_constrints()
        self.set_path()
        self.set_vasicek()
    
    def set_function(self):
        # create listbox for mode
        var = tk.StringVar()
        var.set(("CALI", "SHOW", "ESG"))
        listbox_mode = tk.Listbox(self.frameOne, listvariable=var, height=5, exportselection=False)
        listbox_mode.select_set(0)
        
        tk.Label(self.frameOne, text="Mode", height=1, width=7).place(x=0, y=10)
        listbox_mode.place(x=10, y=40)
        
        # create listbox for algo
        var = tk.StringVar()
        var.set(("Powell", "SLSQP", "trust-constr"))
        listbox_algo = tk.Listbox(self.frameOne, listvariable=var, height=5, exportselection=False)
        listbox_algo.select_set(0)
        
        tk.Label(self.frameOne, text="Algo", height=1, width=7).place(x=200, y=10)
        listbox_algo.place(x=215, y=40)
        
        # create text for core number
        tk.Label(self.frameOne, text="Core numbers", height=1, width=13).place(x=0, y=150)
        txt_core = tk.Text(self.frameOne, height=1, width=10)
        txt_core.insert(INSERT, 8)
        txt_core.place(x=10, y=170)
        
        # create text for bond type
        tk.Label(self.frameOne, text="Bond type", height=1, width=10).place(x=100, y=150)
        txt_btype = tk.Text(self.frameOne, height=1, width=10)
        txt_btype.insert(INSERT, "CORP")
        txt_btype.place(x=110, y=170)
        
        self.listbox_mode = listbox_mode
        self.listbox_algo = listbox_algo
        self.txt_core = txt_core
        self.txt_btype = txt_btype
    
        
    def set_constrints(self):
        tk.Label(self.frameOne, text="l_bound       u_bound              n", height=1, width=25).place(x=450, y=10)
        tk.Label(self.frameOne, text="alpha", height=1, width=4).place(x=400, y=35)
        tk.Label(self.frameOne, text="sigma", height=1, width=4).place(x=400, y=60)
        tk.Label(self.frameOne, text="pi0", height=1, width=2).place(x=400, y=85)
        tk.Label(self.frameOne, text="mu", height=1, width=2).place(x=400, y=110)
        tk.Label(self.frameOne, text="rec", height=1, width=3).place(x=400, y=135)
        
        txt_l_alpha = tk.Text(self.frameOne, height=1, width=5)
        txt_u_alpha = tk.Text(self.frameOne, height=1, width=5)
        txt_n_alpha = tk.Text(self.frameOne, height=1, width=5)
        
        txt_l_sigma = tk.Text(self.frameOne, height=1, width=5)
        txt_u_sigma = tk.Text(self.frameOne, height=1, width=5)
        txt_n_sigma = tk.Text(self.frameOne, height=1, width=5)
        
        txt_l_pi0 = tk.Text(self.frameOne, height=1, width=5)
        txt_u_pi0 = tk.Text(self.frameOne, height=1, width=5)
        txt_n_pi0 = tk.Text(self.frameOne, height=1, width=5)
        
        txt_l_mu = tk.Text(self.frameOne, height=1, width=5)
        txt_u_mu = tk.Text(self.frameOne, height=1, width=5)
        txt_n_mu = tk.Text(self.frameOne, height=1, width=5)
        
        txt_l_rec = tk.Text(self.frameOne, height=1, width=5)
        txt_u_rec = tk.Text(self.frameOne, height=1, width=5)
        txt_n_rec = tk.Text(self.frameOne, height=1, width=5)
        
        txt_l_alpha.insert(INSERT, 0)
        txt_l_alpha.place(x=460, y=35)
        txt_u_alpha.insert(INSERT, 1)
        txt_u_alpha.place(x=525, y=35)
        txt_n_alpha.insert(INSERT, 1)
        txt_n_alpha.place(x=595, y=35)
        
        txt_l_sigma.insert(INSERT, 0)
        txt_l_sigma.place(x=460, y=60)
        txt_u_sigma.insert(INSERT, 5)
        txt_u_sigma.place(x=525, y=60)
        txt_n_sigma.insert(INSERT, 3)
        txt_n_sigma.place(x=595, y=60)
        
        txt_l_pi0.insert(INSERT, 0)
        txt_l_pi0.place(x=460, y=85)
        txt_u_pi0.insert(INSERT, 3)
        txt_u_pi0.place(x=525, y=85)
        txt_n_pi0.insert(INSERT, 1)
        txt_n_pi0.place(x=595, y=85)
        
        txt_l_mu.insert(INSERT, 0)
        txt_l_mu.place(x=460, y=110)
        txt_u_mu.insert(INSERT, 5)
        txt_u_mu.place(x=525, y=110)
        txt_n_mu.insert(INSERT, 2)
        txt_n_mu.place(x=595, y=110)
        
        txt_l_rec.insert(INSERT, 0) #0.38
        txt_l_rec.place(x=460, y=135)
        txt_u_rec.insert(INSERT, 0) #0.55
        txt_u_rec.place(x=525, y=135)
        txt_n_rec.insert(INSERT, 1)
        txt_n_rec.place(x=595, y=135)
        
        
        self.txt_l_alpha, self.txt_u_alpha, self.txt_n_alpha = txt_l_alpha, txt_u_alpha, txt_n_alpha
        self.txt_l_sigma, self.txt_u_sigma, self.txt_n_sigma = txt_l_sigma, txt_u_sigma, txt_n_sigma
        self.txt_l_pi0, self.txt_u_pi0, self.txt_n_pi0 = txt_l_pi0, txt_u_pi0, txt_n_pi0
        self.txt_l_mu, self.txt_u_mu, self.txt_n_mu = txt_l_mu, txt_u_mu, txt_n_mu
        self.txt_l_rec, self.txt_u_rec, self.txt_n_rec = txt_l_rec, txt_u_rec, txt_n_rec
        
    def set_path(self):
        # set result path
        tk.Label(self.frameOne, text="Para path", height=1, width=7).place(x=300, y=200)
        txt_path = tk.Text(self.frameOne, height=1, width=40)
        txt_path.insert(INSERT, "JLT/Result/calibration/CONSTR/CORP_AA_BBBweight(3-7)=10_10_rec=0_parameter.csv")
        txt_path.place(x=300, y=220)
        self.txt_path = txt_path
        
    def set_weight(self):
        tk.Label(self.frameOne, text="Ratings. eg:['AAA', 'BB']", height=1, width=20).place(x=0, y=200)
        txt_rating = tk.Text(self.frameOne, height=1, width=30)
        tk.Label(self.frameOne, text="Years. eg:[[3, 4, 5], [3, 4, 5]]", height=1, width=22).place(x=0, y=250)
        txt_year = tk.Text(self.frameOne, height=1, width=30)
        tk.Label(self.frameOne, text="Values. eg:[[10, 10, 10], [10, 10, 10]]", height=1, width=28).place(x=0, y=300)
        txt_value = tk.Text(self.frameOne, height=1, width=50)

        txt_rating.insert(INSERT, "['AA', 'BBB']")
        txt_rating.place(x=10, y=220)
        txt_year.insert(INSERT, "[[3,4,5,6,7], [3,4,5,6,7]]")
        txt_year.place(x=10, y=270)
        txt_value.insert(INSERT, "[[10,10,10,10,10], [10,10,10,10,10]]")
        txt_value.place(x=10, y=320)
        
        self.txt_rating, self.txt_year, self.txt_value = txt_rating, txt_year, txt_value
    
    
    # *********************** Vasicek part ***************************
    def set_vasicek(self):
        # set mode
        var_mode = tk.StringVar()
        var_mode.set(("CALI", "ESG", "SHOW"))
        vasicek_listbox_mode = tk.Listbox(self.frameTwo, listvariable=var_mode, height=3, exportselection=False)
        vasicek_listbox_mode.selection_set(0)
        
        tk.Label(self.frameTwo, text="Mode", height=1, width=7).place(x=0, y=10)
        vasicek_listbox_mode.place(x=10, y=40)
        self.vasicek_listbox_mode = vasicek_listbox_mode
        
        # set method
        var_method = tk.StringVar()
        var_method.set(("ols", "ml", "ltq"))
        vasicek_listbox_method = tk.Listbox(self.frameTwo, listvariable=var_method, height=3, exportselection=False)
        vasicek_listbox_method.selection_set(0)
        
        tk.Label(self.frameTwo, text="Method", height=1, width=7).place(x=200, y=10)
        vasicek_listbox_method.place(x=215, y=40)
        self.vasicek_listbox_method = vasicek_listbox_method
        
        
    def apply_setting(self):
        config = Config("settings.toml") 
        # set mode
        mode_index = self.listbox_mode.curselection()[0]
        config.get_section("main").set(mode=self.listbox_mode.get(mode_index))
        
        # set algo
        algo_index = self.listbox_algo.curselection()[0]
        config.get_section("calibrate").set(algo=self.listbox_algo.get(algo_index))
        
        # set core number
        config.get_section("main").set(cores=int(self.txt_core.get("1.0",'end-1c')))
        
        # set bond type
        config.get_section("calibrate").set(bond_type=self.txt_btype.get("1.0",'end-1c'))
        
        # set path
        config.get_section("path").set(cali_result_path=self.txt_path.get("1.0",'end-1c'))
        config.get_section("path").set(esg_paras_path=self.txt_path.get("1.0",'end-1c'))
         
        # set weight
        config.get_section("calibrate").set(weight_rate_list=ast.literal_eval(self.txt_rating.get("1.0",'end-1c')))
        config.get_section("calibrate").set(weight_year_list=ast.literal_eval(self.txt_year.get("1.0",'end-1c')))
        config.get_section("calibrate").set(weight_val_list=ast.literal_eval(self.txt_value.get("1.0",'end-1c')))
        
        # set constraints
        config.get_section("calibrate").set(l_alpha=float(self.txt_l_alpha.get("1.0",'end-1c')))
        config.get_section("calibrate").set(u_alpha=float(self.txt_u_alpha.get("1.0",'end-1c')))
        config.get_section("calibrate").set(n_alpha=int(self.txt_n_alpha.get("1.0",'end-1c')))
        
        config.get_section("calibrate").set(l_sigma=float(self.txt_l_sigma.get("1.0",'end-1c')))
        config.get_section("calibrate").set(u_sigma=float(self.txt_u_sigma.get("1.0",'end-1c')))
        config.get_section("calibrate").set(n_sigma=int(self.txt_n_sigma.get("1.0",'end-1c')))
        
        config.get_section("calibrate").set(l_pi_0=float(self.txt_l_pi0.get("1.0",'end-1c')))
        config.get_section("calibrate").set(u_pi_0=float(self.txt_u_pi0.get("1.0",'end-1c')))
        config.get_section("calibrate").set(n_pi_0=int(self.txt_n_pi0.get("1.0",'end-1c')))
        
        config.get_section("calibrate").set(l_mu=float(self.txt_l_mu.get("1.0",'end-1c')))
        config.get_section("calibrate").set(u_mu=float(self.txt_u_mu.get("1.0",'end-1c')))
        config.get_section("calibrate").set(n_mu=int(self.txt_n_mu.get("1.0",'end-1c')))
        
        config.get_section("calibrate").set(l_rec=float(self.txt_l_rec.get("1.0",'end-1c')))
        config.get_section("calibrate").set(u_rec=float(self.txt_u_rec.get("1.0",'end-1c')))
        config.get_section("calibrate").set(n_rec=int(self.txt_n_rec.get("1.0",'end-1c')))
        
        # set vasicek
        vasicek_mode_index = self.vasicek_listbox_mode.curselection()[0]
        config.get_section("vasicek").set(mode=self.vasicek_listbox_mode.get(vasicek_mode_index))
        
        vasicek_method_index = self.vasicek_listbox_method.curselection()[0]
        config.get_section("vasicek").set(method=self.vasicek_listbox_method.get(vasicek_method_index))

    
    def run_main_frame1(self):
        print("Applying settings")
        self.apply_setting()
        print("Start main.py")
        os.system("conda activate && python JLT/main.py")
    
    def run_main_frame2(self):
        print("Applying settings")
        self.apply_setting()
        print("Start main.py")
        os.system("conda activate && python JLT/vasicek.py")
        
    def run(self):
        button_frame1 = tk.Button(self.frameOne, text='run', command=self.run_main_frame1, height=2, width=5)
        # put button at the bottom of window 
        button_frame1.pack(side="bottom")
        
        button_frame2 = tk.Button(self.frameTwo, text='run', command=self.run_main_frame2, height=2, width=5)
        # put button at the bottom of window 
        button_frame2.pack(side="bottom")
        
        self.window.mainloop()

win = Windows().run()
