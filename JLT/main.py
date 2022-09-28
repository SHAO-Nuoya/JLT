'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Tuesday, May 31st 2022, 4:53:54 pm
   Last Update:    Monday, June 27th 2022, 10:43:25 am
   File:           main.py
   Copyright (c) 2022 Allianz
'''

# todo prix de zero coupon risque
from diffuse import Diffuse
from utility import save_calibration_result, complete_esg
from input import SimulationRiskPrime
import itertools
from calibrate import Calibrate
from multiprocessing import Pool
import numpy as np
import pandas as pd
from config import settings
import sys
sys.path.append("C:\\Users\\SHAO\\Desktop\\JLT_Projet")


def target_calibrate(i: int, init_paras: list[float]) -> np.array:
    """get optimal parameters

    Args:
        i (int): subprocess id
        init_paras (list[float]): alpha, sigma, pi_0, mu, rec_A, rec_BC

    Returns:
        np.array: optimal parameters and corresponding loss : alpha, sigma, pi_0, mu, rec_A, rec_BC, loss
    """
    print("Start subprocess ", i)

    res = None
    while res is None:
        Cali = Calibrate()
        try:
            res = Cali.get_parameters(init_paras)
        except ValueError:
            print("Warning : negative parameter encountered, restart the optimization")
            pass
    return res


def target_esg(dTs: list[float], N_scenario_start: int, N_scenario_end: int, N: int, timestep: int, all_dists):
    """create ESG

    Args:
        dTs (list[float]): time to maturity list
        N_start (int): begin index of simulation (included)
        N_end (int): end index of simulation (not included)
        N (int): total simulation number
        timestep (int): maximum projection time

    Returns:
        pd.DataFrame: esg from simulation (N_start) to simulation (N_end-1)
    """
    return Diffuse(settings.path.esg_paras_path).generate_ESG(dTs, N_scenario_start, N_scenario_end, N, timestep, all_dists)


if __name__ == "__main__":
    p = Pool(settings.main.cores)
    if settings.main.mode == 'CALI':
        init_paras = list(itertools.product(np.linspace(settings.calibrate.l_alpha, settings.calibrate.u_alpha, settings.calibrate.n_alpha+2)[1:-1],
                                            np.linspace(
                                                settings.calibrate.l_sigma, settings.calibrate.u_sigma, settings.calibrate.n_sigma+2)[1:-1],
                                            np.linspace(
                                                settings.calibrate.l_pi_0, settings.calibrate.u_pi_0, settings.calibrate.n_pi_0+2)[1:-1],
                                            np.linspace(
                                                settings.calibrate.l_mu, settings.calibrate.u_mu, settings.calibrate.n_mu+2)[1:-1],
                                            np.linspace(
                                                settings.calibrate.l_rec, settings.calibrate.u_rec, settings.calibrate.n_rec+2)[1:-1])
                          )

        res = p.starmap(target_calibrate, [(i, init_paras[i])
                        for i in range(len(init_paras))])
        save_calibration_result(res)
        Diffuse(settings.path.cali_result_path).plot_trajectory(0)

    elif settings.main.mode == 'SHOW':
        Diffuse(settings.path.cali_result_path).plot_trajectory(0)
    elif settings.main.mode == 'ESG':
        N = settings.esg.N//2
        N_scenario_starts = list(range(1, N+1, N//settings.main.cores))
        N_scenario_ends = N_scenario_starts[1:]+[N+1]

        SRP = SimulationRiskPrime(
            Diffuse(settings.path.esg_paras_path).get_best_paras(), settings.esg.seed)

        len_all_dists = settings.constant.DISCERT_DT_N * (int(max(settings.esg.dTs))+settings.esg.timestep)*(N+1)
        if settings.calibrate.schema in ['QE', 'INVERSE']:
            all_dists = np.array(SRP.get_distrubution_sample(
                "unif", len_all_dists))
        else:
            all_dists = np.array(SRP.get_distrubution_sample(
                "norm", len_all_dists))
        
        paras_list = [(settings.esg.dTs, N_scenario_starts[i], N_scenario_ends[i], N,
                       settings.esg.timestep, all_dists) for i in range(len(N_scenario_starts))]
        [print(p[:-1]) for p in paras_list]

        esg = p.starmap(target_esg, paras_list)
        res_spread = [e[0] for e in esg] + [e[1] for e in esg]
        res_prob = [e[2] for e in esg] + [e[3] for e in esg]

        spread_data = pd.concat(res_spread, axis=0)
        prob_data = pd.concat(res_prob, axis=0)

        complete_esg(spread_data, prob_data)
        print("Finished")
    p.close()
    
#dTs = [ 0.0833, 0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,]
