'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Tuesday, July 19th 2022, 9:40:39 am
   Last Update:    Wednesday, July 20th 2022, 10:57:18 am
   File:           vasicek.py
   Copyright (c) 2022 Allianz
'''

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import exp, sqrt, log
from utility import MRG32k3a
from config import settings
from tqdm import tqdm

from config import settings

class Vasicek:
    def __init__(self, paras) -> None:
        """
        Args:
            r0 (float, optional): initial interest rate. Defaults to 0.5.
            T (float, optional): maturity. Defaults to 252 days.
        """
        self.x0, self.alpha, self.theta, self.sigma = paras
        self.random = MRG32k3a(int(datetime.now().timestamp()))

    def generate_process(self, Ts, dists) -> list[float]:
        """
        Args:
            ts (list[float]): time list (unit is year)
        Returns:
            list: list of interest rate for every day
        """
        alpha, theta, sigma, x0 = self.alpha, self.theta, self.sigma, self.x0
        res = [x0]
        dt = 1/settings.constant.DISCERT_DT_N
        for rand in dists:
            x_pre = res[-1]
            b = exp(-alpha*dt)
            c = theta*(1-b)
            delta = sigma*sqrt((1-b**2)/(2*alpha))
            x_new = c+b*x_pre+delta*rand
            res.append(x_new)
        
        res = [res[t*settings.constant.DISCERT_DT_N] for t in Ts]
        return res


class Calibrate:
    def __init__(self) -> None:
        self.data = np.array([1.90, 1.61, 2.56, 2.14, 2.45, 2.16, 2.34, 2.72, 2.44, 3.41, 2.60, 1.85, 
                     2.11, 2.16, 1.83, 1.17, 1.42, 1.23, 1.50, 1.50, 2.06, 1.54, 1.52])/100
    
    def OLS(self):
        data = self.data
        n = len(data)-1
        DT = 1
        
        b_ = (n*sum(data[1:]*data[:-1])-sum(data[1:])*sum(data[:-1]))/(n*sum(data[:-1]**2)-sum(data[:-1])**2)
        theta_= sum(data[1:]-b_*data[:-1])/(n*(1-b_))
        delta_ = sqrt(sum((data[1:]-b_*data[:-1]-theta_*(1-b_))**2)/n)
        alpha_ = -log(b_)/DT
        sigma_ = delta_/(sqrt((b_**2-1)*DT/(2*log(b_))))
        
        return [data[0], alpha_, theta_, sigma_]
    
    def ML(self):
        data = self.data
        n = len(data)-1
        DT = 1
        
        Sx = sum(data[:-1])
        Sy = sum(data[1:])
        Sxx = sum(data[:-1]**2)
        Sxy = sum(data[:-1]*data[1:])
        Syy = sum(data[1:]**2)
        
        theta_ = (Sy*Sxx-Sx*Sxy)/(n*(Sxx-Sxy)-(Sx**2-Sx*Sy))
        alpha_ = ((Sxy-theta_*Sx- theta_*Sy+n*theta_**2)/(Sxx-2*theta_*Sx+n*theta_**2))/DT
        a = 1-alpha_*DT
        sigmah2 = (Syy-2*a*Sxy+a**2*Sxx-2*theta_*(1-a)*(Sy-a*Sx)+n*theta_**2*(1-a)**2)/n
        sigma_ = sqrt(sigmah2*2*alpha_/(1-a**2))
        
        return [data[0], alpha_, theta_, sigma_]

    def LTQ(self):
        sigma_ = np.std(self.data)
        q95 = np.quantile(self.data, 0.95)    
        q05 = np.quantile(self.data, 0.05)
        theta_ = (q05+q95)/2
        alpha_ = 2*((1.96*sigma_)/(q95-q05))**2
        
        return [self.data[0], alpha_, theta_, sigma_]
            
        
    def plot_calibration(self, method:str):
        if method == "ols":
            paras = self.OLS()
        elif method == "ml":
            paras = self.ML()
        elif method == "ltq":
            paras = self.LTQ()
        
        vasicek = Vasicek(paras)
        Rand = MRG32k3a(int(datetime.now().timestamp()))
        dists = [Rand.normal() for _ in range(len(self.data)*settings.constant.DISCERT_DT_N)]
        calis = vasicek.generate_process(range(len(self.data)), dists) 
        plt.plot(calis, color="r", label="simulation")
        plt.scatter(range(len(self.data)), self.data, label="real data")
        plt.title(f"x0={paras[0]:.4f}, alpha={paras[1]:.4f}, theta={paras[2]:.4f}, sigma={paras[3]:.4f}")
        plt.legend()
        plt.show()    
    
    def generate_ESG(self, N_scenario, time_step, esg_path, method):
        if method == "ols":
            paras = self.OLS()
        elif method == "ml":
            paras = self.ML()
        elif method == "ltq":
            paras = self.LTQ()
            
        paras[0] = self.data[-1]
        vasicek = Vasicek(paras)
        
        inflation = []
        
        for scenario in tqdm(range(N_scenario//2)):
            Rand = MRG32k3a(scenario+1)
            dists = np.array([Rand.normal() for _ in range(time_step*settings.constant.DISCERT_DT_N)])
            process = vasicek.generate_process(range(1,61), dists)
            inflation.append(self.data[-1])
            inflation.extend(process)
            
            anti_dists = -dists
            anti_process = vasicek.generate_process(range(1,61), anti_dists)
            inflation.append(self.data[-1])
            inflation.extend(anti_process)
        
        plt.show()
        print("Opening esg.csv")
        ESG = pd.read_csv(esg_path)
        print("Modifying esg.csv")
        ESG["Inflation"] = inflation
        print("Saving esg.csv")
        ESG.to_csv(esg_path, index=False)
    
    def generate_average_esg(self, esg_path):
        avg_esg_path = esg_path.replace(".csv", "_avg.csv")
        
        print("Opening esg.csv")
        ESG = pd.read_csv(esg_path)
        avg_ESG = pd.read_csv(avg_esg_path)
        
        print("Modifying esg.csv")
        avg_inflation = ESG["Inflation"].groupby(ESG["Timestep"]).mean().tolist()
        min_inflation = ESG["Inflation"].groupby(ESG["Timestep"]).min().tolist()
        max_inflation = ESG["Inflation"].groupby(ESG["Timestep"]).max().tolist()
        avg_ESG["Inflation"] = avg_inflation
        
        plt.plot(avg_inflation)
        plt.plot(min_inflation)
        plt.plot(max_inflation)
        plt.show()
        plt.savefig(esg_path.replace(esg_path.split("/")[-1], "overview.png"))
        
        print("Saving esg.csv")
        avg_ESG.to_csv(avg_esg_path, index=False)
        
        
        

if __name__ == "__main__":
    if settings.vasicek.mode == 'CALI':
        Calibrate().plot_calibration(settings.vasicek.method)
    elif settings.vasicek.mode == 'SHOW':
        ESG = pd.read_csv(settings.vasicek.esg_path)
        ESG["Inflation"].groupby(ESG["Scenario"]).apply(lambda x:plt.plot(x.tolist()))
        plt.show()
    elif settings.vasicek.mode == 'ESG':
        Calibrate().generate_ESG(settings.vasicek.N, settings.vasicek.timestep, settings.vasicek.esg_path, settings.vasicek.method)
        Calibrate().generate_average_esg(settings.vasicek.esg_path)
