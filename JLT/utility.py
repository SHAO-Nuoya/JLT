'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Monday, June 13th 2022, 9:58:16 am
   Last Update:    Monday, June 27th 2022, 10:44:01 am
   File:           utility.py
   Copyright (c) 2022 Allianz
'''

from cProfile import label
import sys
sys.path.append("C:\\Users\\SHAO\\Desktop\\JLT_Projet")

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, log, sqrt
from numpy.linalg import matrix_power, norm
import pandas as pd
import pylab
import scipy.stats as stats
from scipy.linalg import expm
from scipy.stats import kstest

from config import settings


class MRG32k3a:
    """Class for generating random number based on MRG32k3a"""

    def __init__(self, seed_state: int = 123) -> None:
        """

        Args:
            seed_state (int, optional): random seed. Defaults to 123.
        """
        self.a1 = [0, 1754669720, -3182104042]
        self.m1 = 2**63 - 6645

        self.a2 = [31387477935, 0, -6199136374]
        self.m2 = 2**63 - 21129 

        self.d = self.m1 + 1
        self.NV_MAGICCONST = 4*exp(-0.5)/sqrt(2.0)

        self.delta1 = 1
        self.delta2 = -1
        self.seed_state = seed_state
        self.seed(seed_state)

    def seed(self, seed_state: int) -> None:
        """set random seed

        Args:
            seed_state (int): seed value
        """
        assert 0 < seed_state < self.d, f"Out of Range 0 x < {self.d}"
        self.x1 = [seed_state, 0, 0]
        self.x2 = [seed_state, 0, 0]

    def randint(self) -> int:
        """return random int in range 0..self.d

        Returns:
            int: random integer from 0 to self.d
        """
        x1i = sum(aa * xx for aa, xx in zip(self.a1, self.x1)) % self.m1
        x2i = sum(aa * xx for aa, xx in zip(self.a2, self.x2)) % self.m2
        self.x1 = [x1i] + self.x1[:2]
        self.x2 = [x2i] + self.x2[:2]

        z = (x1i*self.delta1 + x2i*self.delta2) % self.m1
        # answer = (z + 1)
        answer = self.m1 if z == 0 else z
        return answer

    def uniform(self) -> float:
        """return uniform random float between 0 and 1

        Returns:
            float: uniform random float between 0 and 1
        """
        return self.randint() / self.d

    def normal(self, mu: float = 0, sigma: float = 1) -> float:
        """return normal random float between 0 and 1, Kinderman and Monahan method

        Args:
            mu (float, optional): mean. Defaults to 0.
            sigma (float, optional): standard deviation. Defaults to 1.

        Returns:
            float: normal random float
        """
        while True:
            u1 = self.uniform()
            u2 = 1.0-self.uniform()
            z = self.NV_MAGICCONST*(u1-0.5)/u2
            zz = z**2/4.0
            if zz <= -log(u2):
                break

        num = mu+z*sigma
        # num = np.random.normal()
        return num


# todo : Can be improved
def transition_matrix_to_generator(tran_mat: pd.DataFrame, year: float = 1, ite: int = 10) -> pd.DataFrame:
    """transform transition matrix into generator

    Args:
        tran_mat (np.ndarray): original transition matrix
        year (float, optional):  time scale for transition matrix. Defaults to 1.
        ite (int, optional): Iteration times for transformation. Defaults to 10.

    Returns:
        pd.DataFrame: generator matrix
    """
    Q = pd.DataFrame(np.zeros(tran_mat.shape),
                     index=tran_mat.index, columns=tran_mat.columns)
    I = np.eye(Q.shape[0])
    for i in range(ite):
        Q = Q + (-1)**i*matrix_power(tran_mat-I, i+1)/(i+1)

    return Q


def evaluate_transformation(transition_matrix: pd.DataFrame, transition_generator: pd.DataFrame, year: float = 1) -> None:
    """compare the original transition matrix and the one generated from transition generator

    Args:
        transition_matrix (pd.DataFrame): original transition matrix
        transition_generator (pd.DataFrame): transition generator obtained from historical transition matrix
        year (float, optional): time scale for transition matrix. Defaults to 1.
    """
    pseudo_trans_matrix = expm(np.array(transition_generator*year))
    err = norm(np.array(transition_matrix)-pseudo_trans_matrix)
    print(
        f"Difference between generated transition matrix and real transition matrix is \n {pseudo_trans_matrix-transition_matrix}")
    print(
        f"The norm between generated transition matrix and real tranition matrix is {err}")


def check_p_val(p_val: float, alpha: float) -> None:
    """Show statistics test result

    Args:
        p_val (float): p value from statistics test
        alpha (float): significance factor
    """
    if p_val < alpha:
        print('We have evidence to reject the null hypothesis.')
    else:
        print('We do not have evidence to reject the null hypothesis.')


def lattice_structure_test(rand_array: np.array) -> None:
    """Plot the lattice structure for generated uniform distribution

    Args:
        rand_array (np.array): generated data
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("2D Lattice structure")
    axs[0].scatter(rand_array[:-1], rand_array[1:], s=0.1)

    plt.title("3D Lattice structure")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(rand_array[:-2], rand_array[1:-1], rand_array[2:], s=0.1)

    plt.savefig("Result/random_test/lattice_structure.png")
    plt.show()


def randomicity_test(rand_array: np.array, dist: str) -> None:
    """Test distribution of generated data

    Args:
        rand_list (np.array): generated data
        dist (str): distribution name, choose from ["norm", "uniform"]
    """
    xs = np.arange(np.min(rand_array), np.max(rand_array), 0.01)

    if dist == "norm":
        fit = stats.norm.pdf(xs, np.mean(rand_array), np.std(rand_array))
    elif dist == "uniform":
        fit = stats.uniform.pdf(xs, 0, 1)

    stat, p_val = kstest(rand_array, dist, (0, 1))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    stats.probplot(rand_array, dist=dist, plot=pylab)

    plt.subplot(1, 2, 2)
    plt.plot(xs, fit, label=dist+" distribution ", lw=3)
    plt.hist(rand_array, 100, density=True, label='actual data')
    plt.legend(prop={"size": 10})
    plt.savefig(f"JLT/Result/random_test/{dist}.png")
    plt.show()

    print('\nStatistic: \t{:1.2f} \nP-Value: \t{:1.2e}'.format(stat, p_val))
    check_p_val(p_val, alpha=0.05)


def preprocessing(bond_type) -> list:
    """Preprocess historical data \n
       We extract data that verifies several criteriors: \n
       1. Time to maturity is integer year + 1 month \n
       2. Date is the minimum among all possible dates in the file \n

    Args:
        file_path (str): historical spread data path
        type (str): bond type, "CORP", "GOVD"

    Returns:
        pd.dataframe: spread for each state with different years to maturity \n
    """
    df = pd.read_excel(settings.path.market_data_path, usecols=["Date", "Instrument", "Term", "Value"]).dropna(axis=0)

    if bond_type == "CORP":
        max_year = 12
        bond_list = ["EUR-SPREAD-CORP-A", "EUR-SPREAD-CORP-AA", "EUR-SPREAD-CORP-AAA",
                     "EUR-SPREAD-CORP-B", "EUR-SPREAD-CORP-BB", "EUR-SPREAD-CORP-BBB",
                     "EUR-SPREAD-CORP-C", ]  # "EUR-SPREAD-CORP-CC", "EUR-SPREAD-CORP-CCC"]
    elif bond_type == "GOVD":
        max_year = 31
        bond_list = ["EUR-SPREAD-GOVD-A", "EUR-SPREAD-GOVD-AA", "EUR-SPREAD-GOVD-AAA",
                     "EUR-SPREAD-GOVD-B", "EUR-SPREAD-GOVD-BB", "EUR-SPREAD-GOVD-BBB",
                     "EUR-SPREAD-GOVD-C", ]  # "EUR-SPREAD-GOVD-CC", "EUR-SPREAD-GOVD-CCC"]

    term_list = [30, 91]+[365*i for i in range(1, max_year)]
    
    df = df[(df["Term"].isin(term_list)) & (df["Instrument"].isin(bond_list))]
    # We only consider data whose maturity is less than max_year years
    ts = sorted(list(filter(lambda x: x < max_year, set(
        ((df["Term"].values)/365).tolist()))))
    instruments = set(df["Instrument"].values.tolist())

    # key is the instrument name, value is the rate(state)
    states_dict = {ins: ins.split('-')[-1] for ins in instruments}

    spread = {state: np.array([]) for state in states_dict.values()}
    for instru, state in states_dict.items():
        for t in ts:
            sp = df[(df["Instrument"] == instru) & (
                df["Term"] == (t*365))]["Value"].values[0]
            spread[state] = np.append(spread[state], sp)

    spread["year_to_mat"] = ts
    spread = pd.DataFrame.from_dict(spread)
    return spread


def save_calibration_result(res: list[list]) -> None:
    """save calibration result

    Args:
        res (list[list]): alpha, sigma, pi_0, mu, rec, loss
    """
    res_df = pd.DataFrame.from_dict({"alpha": [], "sigma": [], "pi_0": [], "mu": [
    ], "rec": [], "func": []})
    res_df["alpha"] = [re[0] for re in res]
    res_df["sigma"] = [re[1] for re in res]
    res_df["pi_0"] = [re[2] for re in res]
    res_df["mu"] = [re[3] for re in res]
    res_df["rec"] = [re[4] for re in res]
    res_df["func"] = [re[5] for re in res]

    res_df.to_csv(settings.path.cali_result_path)


def plot_calibration_result_distri(loss_threshold=1.6) -> None:
    """plot calibrated parameters' distribution 

    Args:
        loss_threshold (float, optional): select parameters whose error is less than loss_threshold. Defaults to 1.6.
    """
    res_df = pd.read_csv(settings.path.cali_result_path)
    res_df = res_df[res_df["func"] < loss_threshold]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Parameters distribution")

    axs[0, 0].hist(res_df["alpha"], bins=(50))
    axs[0, 0].set_xlabel("alpha")

    axs[0, 1].hist(res_df["sigma"], bins=50)
    axs[0, 1].set_xlabel("sigma")

    axs[1, 0].hist(res_df["pi_0"], bins=50)
    axs[1, 0].set_xlabel("pi_0")

    axs[1, 1].hist(res_df["mu"], bins=50)
    axs[1, 1].set_xlabel("mu")

    plt.savefig(settings.path.cali_result_path.replace(".csv", "_distri_.png"))
    plt.show()

    plt.hist(res_df["func"], bins=50)
    plt.title("loss function distribution")
    plt.xlabel("loss")
    plt.savefig(settings.path.cali_result_path.replace(
        ".csv", "_loss_distri_.png"))
    plt.show()

def plot_esg_spread_distibution(spread_path:str, timestep:int, maturity:str):
    spread_df = pd.read_csv(spread_path)
    # spread_df = spread_df[spread_df["Timestep"]==timestep]
    # for s in settings.esg.states:
    #     col = f"EUR_CORP.{s}.SpreadPrice({maturity})"
    #     plt.hist(spread_df[col], bins=10)
    #     plt.title(s)
    #     plt.savefig(f"JLT/Result/ESG/distribution/{s}_timestep={timestep}_mat={maturity}.png")
    #     plt.show()
    cols = ["Timestep"] + [col for col in spread_df.columns if "EUR_CORP.BBB.SpreadPrice" in col]
    spread_df = spread_df[cols]
    df = spread_df.groupby("Timestep").mean()
    for i in range(df.shape[0]):
        plt.plot(df.iloc[i, :].values)
    plt.show()

def complete_esg(spread_data:pd.DataFrame, prob_data:pd.DataFrame, extra_type = ["FIN", "GOVD"]):
    # complete esg
    spread_path = settings.path.esg_spread_result_path
    prob_path = spread_path.replace("spread", "prob")

    spread_cols = spread_data.columns[3:]
    prob_cols = prob_data.columns[2:]
    
    for extra in extra_type:
        new_spread_cols = list(map(lambda x:x.replace("CORP", extra), spread_cols))
        spread_data["EUR_"+extra+".RecoveryRate"] = 0
        spread_data[new_spread_cols] = 1
        
        # # calculate bond price
        # spread_data[new_spread_cols] = exp(-spread_data[new_spread_cols].values*np.array(settings.esg.dTs))
        
        new_prob_cols = list(map(lambda x:x.replace("CORP", extra), prob_cols))
        prob_data[new_prob_cols] = prob_data[prob_cols]
    
    # complete avg esg
    average_spread_data = spread_data.groupby("Timestep", ).mean()
    average_spread_data["Scenario"] = "AVERAGE"
    average_spread_col = average_spread_data.pop("Scenario")
    average_spread_data.insert(0, average_spread_col.name, average_spread_col)
    average_spread_data.to_csv(settings.path.esg_spread_result_path.replace(".csv", "_avg.csv"), index=False)
    
    average_prob_data = prob_data.groupby("Timestep").mean()
    average_prob_data["Scenario"] = "AVERAGE"
    average_prob_col = average_prob_data.pop("Scenario")
    average_prob_data.insert(0, average_prob_col.name, average_prob_col)
    average_prob_data.to_csv(settings.path.esg_prob_result_path.replace(".csv", "_avg.csv"), index=False)

    spread_data.to_csv(spread_path, index=False)
    prob_data.to_csv(prob_path, index=False)
    
