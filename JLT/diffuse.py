'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Friday, June 17th 2022, 4:56:17 am
   Last Update:    Monday, June 27th 2022, 10:41:58 am
   File:           diffuse.py
   Copyright (c) 2022 Allianz
'''
import pandas as pd
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from tqdm import tqdm

from input import JLT, SimulationRiskPrime
from utility import preprocessing
from config import settings


class Diffuse:
    """diffuse spread with specific parameters
    """

    def __init__(self, path: str) -> None:
        """

        Args:
            path (str): calibration result path
        """
        self.path = path
        self.paras = self.get_best_paras()
        self.jlt = JLT(self.paras)

    def get_best_paras(self):
        df = pd.read_csv(self.path)
        # get parameters with smallest error
        res = df.iloc[df["func"].idxmin(), 1:-1].values.tolist()
        # alpha, sigma, pi_0, mu, rec
        # res = [ 0.1783, 0.00001, 1.6516, 1,  0]
        # res = [ 0.1783, 0.8301, 1.6516, 1,  0]
        return res

    def plot_trajectory(self, t: float):
        """plot spread

        Args:
            t (float): current time (or projection)
        """
        draw_states = settings.common.draw_states
        market_traj = preprocessing(settings.calibrate.bond_type)
        dTs = market_traj["year_to_mat"]

        SRP = SimulationRiskPrime(self.paras, settings.calibrate.seed)

        if settings.calibrate.schema in ['QE', 'INVERSE']:
            dist = "unif"
        else:
            dist = "norm"
        dists = SRP.get_distrubution_sample(
            dist, int(t+max(dTs)*settings.constant.DISCERT_DT_N))
        pis = SRP.generate_CIR_process(settings.calibrate.schema, dists)
        traj = self.jlt.get_spread_theo(
            t, dTs, pis, settings.common.draw_states)

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        # Plot simulated spread and market spread
        for state in draw_states:
            plt.plot(dTs, traj[state], label=state+" cali", linestyle='--')
            plt.scatter(dTs, market_traj[state], label=state+" market", s=20)

        plt.xlabel("year")
        plt.ylabel("spread")
        plt.title(
            f"alpha:{self.paras[0]:.3f}  sigma:{self.paras[1]:.3f}  pi_0:{self.paras[2]:.3f}  mu:{self.paras[3]:.3f}  rec:{self.paras[4]:.3f}")
        plt.legend(loc='upper right')

        # Plot difference between simulated spread and market spread
        plt.subplot(2, 1, 2)
        width = 0.15
        for i, state in enumerate(draw_states):
            y = traj[state] - market_traj[state]
            plt.bar(np.arange(len(dTs))+width*i, y, label=state, width=width)

        plt.xticks(dTs)
        plt.xlabel("year")
        plt.ylabel("simulation - market spread")
        plt.legend()
        plt.savefig(settings.path.cali_result_path.replace("csv", "png"))
        plt.show()

    def normalize(self, prob: np.array):
        prob[prob < 0] = 0
        step = len(settings.esg.states) + 1
        for i in range(len(prob)//step):
            prob[i*step:(i+1)*step] = prob[i*step:(i+1)*step] / \
                         sum(prob[i*step:(i+1)*step])

    def generate_ESG(self, dTs: list[float], N_start: int, N_end: int, N: int, timestep: int, all_dists:list[float]) -> pd.DataFrame:
        """create esg .csv file for probability and spread

        Args:
            dTs (list[float]): time to maturity list
            N_start (int): begin index of simulation (included)
            N_end (int): end index of simulation (not included)
            N (int): total simulation number//2 (reduce variation)
            timestep (int): maximum projection time

        Returns:
            pd.DataFrame: esg from simulation (N_start) to simulation (N_end-1)
        """
        
        # create columns names for spread
        spread_cols = ['Scenario', 'Timestep',
            f"EUR_{settings.esg.bond_type}.RecoveryRate"]
        for state in settings.esg.states:
            spread_cols.extend([f"EUR_{settings.esg.bond_type}.{state}.SpreadPrice(1m)",
                               f"EUR_{settings.esg.bond_type}.{state}.SpreadPrice(3m)"])
            for t in dTs[2:]:
                col = f"EUR_{settings.esg.bond_type}.{state}.SpreadPrice({int(t)}y)"
                spread_cols.append(col)

        # create columns names for probabilité
        prob_cols = ['Scenario', 'Timestep']
        for r1 in ["AAA", "AA", "A", "BBB", "BB", "B", "C"]:
            for r2 in ["AAA", "AA", "A", "BBB", "BB", "B", "C", "D"]:
                prob_cols.append(f"EUR_{settings.esg.bond_type}.{r1}_to_{r2}")

        spread_res, spread_res_antithetic = [], []
        prob_res, prob_res_antithetic = [], []

    
        # same value when timestep = 0
        SRP = SimulationRiskPrime(self.paras, settings.esg.seed)

        for n in tqdm(range(N_start, N_end)):
            len_dist = settings.constant.DISCERT_DT_N*(int(max(settings.esg.dTs))+settings.esg.timestep)
            start_dist = n*len_dist
            dists = all_dists[start_dist : start_dist + len_dist]
  
            pis = SRP.generate_CIR_process(settings.esg.schema, dists, self.paras[2])
            
            if settings.esg.schema in ['QE', 'INVERSE'] :
                pis_antithetic = SRP.generate_CIR_process(settings.esg.schema, 1-dists, self.paras[2])
            else:
                pis_antithetic = SRP.generate_CIR_process(settings.esg.schema, -dists, self.paras[2])
            

            for t in range(timestep+1):
                #************************* ESG probabilité **************************
                # Scenario, timestep
                prob_line, prob_line_antithetic = [n, t], [n+N, t]

                pi_t = pis[t*settings.constant.DISCERT_DT_N]
                pi_t_antithetic = pis_antithetic[t*settings.constant.DISCERT_DT_N]

                prob = self.jlt.trans_proba(pi_t)
                prob_antithetic = self.jlt.trans_proba(pi_t_antithetic)
                self.normalize(prob)
                self.normalize(prob_antithetic)

                prob_line.extend(prob[:])
                prob_line_antithetic.extend(prob_antithetic[:])
                
                prob_res.append(prob_line[:])
                prob_res_antithetic.append(prob_line_antithetic[:])

                #************************* ESG spread **************************
                # recovery rate in ESG
                RR = self.paras[-1]
                
                # Scenario, timestep, recovery rate
                spread_line, spread_line_antithetic = [n, t, RR], [n+N, t, RR]
                
                spread_df = self.jlt.get_spread_theo(
                    t, dTs, pis, settings.esg.states)
                spread_df_antithetic = self.jlt.get_spread_theo(
                    t, dTs, pis_antithetic, settings.esg.states)

                for state in settings.esg.states:
                    state_spread, state_spread_antithetic = spread_df[state], spread_df_antithetic[state]

                    # get bond price
                    state_spread = exp(-state_spread* \
                                       np.array(settings.esg.dTs))
                    state_spread_antithetic = exp(
                        -state_spread_antithetic*np.array(settings.esg.dTs))

                    spread_line.extend(state_spread)
                    spread_line_antithetic.extend(state_spread_antithetic)

                spread_res.append(spread_line)
                spread_res_antithetic.append(spread_line_antithetic)


        return [pd.DataFrame(spread_res, columns=spread_cols),
                pd.DataFrame(spread_res_antithetic, columns=spread_cols),
                pd.DataFrame(prob_res, columns=prob_cols),
                pd.DataFrame(prob_res_antithetic, columns=prob_cols)]

        
