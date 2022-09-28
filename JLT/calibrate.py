'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Monday, June 13th 2022, 10:09:18 am
   Last Update:    Monday, June 27th 2022, 10:49:59 am
   File:           calibrate.py
   Copyright (c) 2022 Allianz
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize, shgo, Bounds

from config import settings
from input import JLT, SimulationRiskPrime
from utility import preprocessing

class Calibrate:
    def __init__(self) -> None:
        """Calibrate JLT
        """
        self.ite_num = 0


    def get_spread_market(self)->np.ndarray:
        """get historical spread in the market

        Returns:
            np.ndarray: historical spread for each state with different years to maturity
        """
        bond_type = settings.calibrate.bond_type
        if " " in bond_type:
            types = bond_type.split(" ")
            spread_df1 = preprocessing(types[0])
            spread_df2 = preprocessing(types[1])
            spread_df = pd.concat(spread_df1, spread_df2, axis=1, fill=None)
        
        else:
            spread_df = preprocessing(bond_type)
        
        self.dTs = spread_df["year_to_mat"].values.tolist()
        return spread_df[settings.calibrate.states+["year_to_mat"]]


    def get_spread_theo(self, t:float, paras: list[float]) -> np.array:
        """get theoretical spread from JLT model

        Args:
            t (float): current time (or projection)
            paras (list[float]): alpha, sigma, pi_0, mu, rec

        Returns:
            np.array: i-th row, j-th column means spread of rating i bond with j year maturity
        """
        SRP = SimulationRiskPrime(paras, settings.calibrate.seed)
        
        if settings.calibrate.schema == 'QE':
            dist = "unif"
        else:
            dist = "norm"
        dists = SRP.get_distrubution_sample(dist, int(t+max(self.dTs)*settings.constant.DISCERT_DT_N))
        pis = SRP.generate_CIR_process(settings.calibrate.schema, dists)
        res = JLT(paras).get_spread_theo(t, self.dTs, pis, settings.calibrate.states)
        return res

    # todo : How to use moments method to initialize parameters
    def get_init_parameters(self, N=10000)->np.array:
        """Use moment method to guess initial values of parameters 

        Args:
            N (int, optional): Iteration number for moment method. Defaults to 10000.

        Returns:
            np.array: initial guess for alpha, sigma, pi_0, mu
        """
        init_alpha = 0.5
        init_sigma = 0.5
        init_pi_0 = 1
        init_mu = 1
        init_rec = 0.3
        return np.array([init_alpha, init_sigma, init_pi_0, init_mu, init_rec])

    
    def set_weight(self, df:pd.DataFrame)->None:
        """set weight to loss

        Args:
            df (pd.DataFrame): original loss dataframe
        """
        weight_df = pd.DataFrame(np.ones(df.shape), index=df.index, columns=df.columns)
        for i, state in enumerate(settings.calibrate.weight_rate_list):
            for year, val in zip(settings.calibrate.weight_year_list[i], settings.calibrate.weight_val_list[i]):
                weight_df.loc[year, state] = val
        
        df *= weight_df
    
    
    def loss_function(self, para:np.array)->float:
        """Loss function used for minimization

        Args:
            para (np.array[float]): alpha, sigma, pi_0, mu, rec

        Returns:
            float: loss value which should be minimized
        """
        self.ite_num += 1
        diff_df = self.get_spread_market()-self.get_spread_theo(settings.calibrate.t, para)
        diff_df["year_to_mat"] = self.get_spread_market()["year_to_mat"]
        diff_df.set_index("year_to_mat", inplace=True)
        
        if settings.calibrate.weight_rate_list:
            self.set_weight(diff_df)
        
        res = np.sum(np.array(diff_df)**2)#+(sigma-init_sigma)**2+(alpha-init_alpha)**2+(pi_0-mu)**2+(pi_0-init_pi_0)**2

        self.callback(para, res)
        return res

    def callback(self, x:np.array, res:float)->None:
        """callback function for optimize

        Args:
            x (np.array): parameters values after each iteration
            res (float): loss function value
        """
        print(f"Ite:{self.ite_num}  alpha:{x[0]:.8f}  sigma:{x[1]:.8f}  pi_0:{x[2]:.8f}  mu:{x[3]:.8f}  rec:{x[4]:.8f}  loss:{res:.8f}")
    
    
    def get_parameters(self, init_paras:list[float])->np.array:
        """Use optimizer to minimize loss funtion and get optimal parameters

        Args:
            init_paras (list[float]): initial parameters for alpha, sigma, pi_0, mu, rec
        Returns:
            np.array: alpha, sigma, pi_0, mu, rec, loss
        """
        lbounds = [settings.calibrate.l_alpha, settings.calibrate.l_sigma, settings.calibrate.l_pi_0, settings.calibrate.l_mu, settings.calibrate.l_rec]
        ubounds = [settings.calibrate.u_alpha, settings.calibrate.u_sigma, settings.calibrate.u_pi_0, settings.calibrate.u_mu, settings.calibrate.u_rec]
        
        if settings.calibrate.algo == "Powell":
            res = minimize(self.loss_function,
                        init_paras,
                        options={'disp': True, "xtol":0.1,"ftol":0.1},
                        bounds=Bounds(np.array(lbounds), np.array(ubounds)), 
                        method="Powell")
        elif settings.calibrate.algo == "SLSQP":
            ubounds[-1] = ubounds[-1] + 0.0000001
            res = minimize(self.loss_function,
                        init_paras,
                        options={'disp': True},
                        # default ftol': 1e-06
                        bounds=Bounds(np.array(lbounds), np.array(ubounds)), 
                        method="SLSQP")
        
        elif settings.calibrate.algo == "trust-constr":
            res = minimize(self.loss_function,
                        init_paras,
                        options={'verbose': 1, "xtol":0.0001,"gtol":0.0001, "maxiter":30},
                        bounds=Bounds(np.array(lbounds), np.array(ubounds)), 
                        method='trust-constr')

        paras = res.x
        print(paras)
        return np.append(paras, res.fun)




    