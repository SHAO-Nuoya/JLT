'''
   Version:        0.1
   Author:         SHAO Nuoya, nuoya.shao@allianz.fr
   Creation Date:  Tuesday, May 24th 2022, 3:03:38 am
   Last Update:    Monday, June 27th 2022, 10:42:54 am
   File:           input.py
   Copyright (c) 2022 Allianz
'''

from utility import MRG32k3a, transition_matrix_to_generator
from config import settings

import numpy as np
from numpy import exp, sqrt, log
from scipy import stats
from scipy.special import comb
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt
from scipy.stats import ncx2

@jit
def generate_brownien_motion(T: float, N: int) -> np.ndarray:
    """generate a sequence of brownien motion 

    Args:
        T (float): time scale
        N (int): time step number 

    Returns:
        np.ndarray: brownien motion array with shape (n,1) 
    """
    dt = T/N
    BMs = np.zeros(1)
    random = MRG32k3a(datetime.now().timestamp())
    for _ in range(N-1):
        BM = BMs[-1] + sqrt(dt)*random.normal()
        BMs = np.append(BMs, BM)

    return np.array(BMs).reshape(-1, 1)


class JLT:
    def __init__(self, para: list[float]) -> None:
        """

        Args:
            para (list[float]): alpha, sigma, pi_0, mu, rec
        """
        self.states = settings.common.all_states
        self.para = para

    def get_hist_transition_matrix(self) -> pd.DataFrame:
        """set historical transition matrix from Allianz

        Returns:
            pd.DataFrame: historical transition matrix
        """
        matrix = pd.read_csv(
            settings.path.transition_matrix_path).set_index("Unnamed: 0")
        matrix.index.name = "Index"
        return matrix

    def get_hist_transition_generator(self, year: float = 1) -> pd.DataFrame:
        """

        Args:
            year (float, optional): transition matrix time unit. Defaults to 1.

        Returns:
            pd.DataFrame: historical transition generator
        """
        return transition_matrix_to_generator(self.get_hist_transition_matrix(), year=year)

    def trans_proba(self, pi_t:float, dT: float=1):
        """calculate unconditional default probability for different state with different time to maturity 

        Args:
            t (float): current year (or projection year)  
            dTs (list[float]): maturity list (year)
            pis (list[float]): simulated risk prime
        Returns:
            dict: default probability with key:state, value:list of proba for different years to maturity
        """
        alpha, sigma, _, mu, _ = self.para

        hist_generator = np.array(
            self.get_hist_transition_generator(), dtype=float)
        d, Sigma = np.linalg.eig(hist_generator)

        P = []
        for i in range(len(self.states)-1):
            # pi is the proba list from i to all the other 
            for j in range(len(self.states)):
                # res is the default proba of state at t
                prob = 0
                # Calculate Pi,K(t, T)
                for k in range(len(self.states)-1):
                    vj = sqrt(alpha**2-2*d[k]*sigma**2)
                    Aterm1 = 2*alpha*mu/sigma**2
                    Aterm2 = 2*vj*exp((alpha+vj)*dT/2)
                    Aterm3 = (alpha+vj)*(exp(vj*dT)-1)+2*vj
                    Bterm1 = -2*d[k]*(exp(vj*dT)-1)
                    Bterm2 = Aterm3
                    Aj = Aterm1*log(Aterm2/Aterm3)
                    Bj = Bterm1/Bterm2

                    prob += Sigma[i, k] * np.linalg.inv(Sigma)[k, j]*(
                        exp(Aj-pi_t*Bj)-1)
                prob = 1+prob if i==j else prob
                P.append(max(prob, 0))
        return np.array(P)
    
    def default_proba(self, t: float, dTs: list[float], pis: list[float]) -> dict:
        """calculate unconditional default probability for different state with different time to maturity 

        Args:
            t (float): current year (or projection year)  
            dTs (list[float]): maturity list (year)
            pis (list[float]): simulated risk prime
        Returns:
            dict: default probability with key:state, value:list of proba for different years to maturity
        """
        alpha, sigma, _, mu, _ = self.para

        hist_generator = np.array(
            self.get_hist_transition_generator(), dtype=float)
        d, Sigma = np.linalg.eig(hist_generator)

        default_P = {}
        for i_state, state in enumerate(self.states[:-1]):
            # P_state is the default proba list for one state
            P_state = []
            for dT in dTs:
                # res is the default proba of state at t
                res = 0
                # Calculate Pi,K(t, T)
                for j in range(len(self.states)-1):
                    vj = sqrt(alpha**2-2*d[j]*sigma**2)
                    Aterm1 = 2*alpha*mu/sigma**2
                    Aterm2 = 2*vj*exp((alpha+vj)*dT/2)
                    Aterm3 = (alpha+vj)*(exp(vj*dT)-1)+2*vj
                    Bterm1 = -2*d[j]*(exp(vj*dT)-1)
                    Bterm2 = Aterm3
                    Aj = Aterm1*log(Aterm2/Aterm3)
                    Bj = Bterm1/Bterm2

                    res += Sigma[i_state, j] * np.linalg.inv(Sigma)[j, -1]*(
                        exp(Aj-pis[t*settings.constant.DISCERT_DT_N]*Bj)-1)

                P_state.append(max(res, 0))
            default_P[state] = np.array(P_state)
            default_P["DEFAULT"] = np.array([1 for _ in range(len(dTs))])
        return default_P

    def plot_default_prob(self, t:float, dTs: list[float] = range(11)) -> None:
        """Plot default probability for different ratings

        Args:
            t (float): current year (or projection year)  
            dTs (list[float], optional): years to maturity list. Defaults to range(11).
        """
        SRP = SimulationRiskPrime(self.para, settings.calibrate.seed)
        dists = SRP.get_distrubution_sample("unif", int(t+max(dTs))*settings.constant.DISCERT_DT_N)
        pis = SRP.generate_CIR_process("QE", dists)
        
        default_P = self.default_proba(t, dTs, pis)
        for state in default_P.keys():
            plt.plot(dTs, default_P[state], label=state)
        plt.title(f"Default probability")
        plt.xlabel("Maturity")
        plt.legend()
        plt.show()

    def get_spread_theo(self, t: float, dTs: list[float], pis: list[float], states: list[str]) -> pd.DataFrame:
        """get theoretical spread from JLT model

        Args:
            t (float): current year (or projection year)  
            dTs (list[float]): maturity list (year)
            pis (list[float]): simulated risk prime
            states (list[str]): bond ratings list

        Returns:
            pd.DataFrame: i-th row, j-th column means spread of rating i bond at time j
        """
        default_P = self.default_proba(t, dTs, pis)
        rec = self.para[-1]
        # todo  vectorize
        array = np.zeros((len(dTs), len(states)))
        for j, state in enumerate(states):
            for i, dT in enumerate(dTs):
                array[i, j] = -log(1-(1-rec) *
                                   default_P[state][i])/dT

        df = pd.DataFrame(array, index=range(len(dTs)), columns=states)
        df["year_to_mat"] = dTs

        return df


class SimulationRiskPrime:
    def __init__(self, para: list[float], seed: int = 12345678) -> None:
        """

        Args:
            para (list[float]): alpha, sigma, pi_0, mu, rec
            seed (int, optional): random seed. Defaults to 12345678.
        """
        self.para = para
        self.random = MRG32k3a(seed)

    def get_distrubution_sample(self, dist:str, N:int)->list[float]:
        """get samples of distribution of size N

        Args:
            dist (str): distribution name, unif or norm
            N (int): number of samples
        """
        samples = []
        
        if dist == "unif":
            for _ in range(N):
                samples.append(self.random.uniform())
        elif dist == "norm":
            for _ in range(N):
                samples.append(self.random.normal())

        return samples
        
        
    def generate_CIR_process(self, schema: str, dists:list[float], pi_0:float=-1) -> list[float]:
        """one time step discretization for risk prime

        Args:
            schema (str, optional): method to discretize. Defaults to "QE".
            dists (list[float]): simulated samples of specific distribution, uniform for "QE", normal for the other 

        Returns:
            list[float]: CIR process, length = len(dists) + 1
        """
        
        dt = 1/settings.constant.DISCERT_DT_N
        if pi_0 == -1:
            alpha, sigma, pi_0, mu, _ = self.para
        else:
            alpha, sigma, _, mu, _ = self.para
            
        CIRs = [pi_0]
        
        if schema == "QE":
            phi_c = 1.5
            exp_adt = exp(-alpha*dt)
            
            for i in range(len(dists)):
                m = mu+(CIRs[-1]-mu)*exp_adt
                s2 = CIRs[-1]*sigma**2*exp_adt/alpha * \
                    (1-exp_adt)+mu*sigma**2/(2*alpha)*(1-exp_adt)**2

                phi = s2/m**2
                Uv = dists[i]
                if phi <= phi_c:
                    b = sqrt(2/phi-1+sqrt(2/phi*(2/phi-1)))
                    a = m/(1+b**2)
                    Zv = stats.norm(0, 1).ppf(Uv)
                    CIRs.append(a*(b+Zv)**2)
                else:
                    p = (phi-1)/(phi+1)
                    beta = (1-p)/m
                    if Uv <= p:
                        CIRs.append(0)
                    else:
                        CIRs.append(log((1-p)/(1-Uv))/beta)
        elif schema == "ABS":
            for i in range(len(dists)):
                if CIRs[-1] < 0:
                    CIRs[-1] = abs(CIRs[-1])
                CIRs.append(CIRs[-1]+alpha*(mu-CIRs[-1])*dt+sigma*sqrt(CIRs[-1])*sqrt(dt)*dists[i])
        elif schema == "ZERO":
            for i in range(len(dists)):
                if CIRs[-1] < 0:
                    CIRs[-1] = 0
                CIRs.append(CIRs[-1]+alpha*(mu-CIRs[-1])*dt+sigma*sqrt(CIRs[-1])*sqrt(dt)*dists[i])
        elif schema == "INVERSE":
            df = 4*alpha*mu/sigma**2
            c = 2*alpha/((1-exp(-alpha*dt))*sigma**2)
            for i in range(len(dists)):
                cir_pre = CIRs[-1]
                nc = 2*c*cir_pre*exp(-alpha*dt)
                Y = ncx2.ppf(dists[i], df, nc)
                CIRs.append(Y/(2*c))
        return CIRs


    def theo_CIR_moment(self, n: int, t: float) -> float:
        """Calculate theoretical moments for CIR process

        Args:
            n (int): Moment order
            t (float): year

        Returns:
            float: Desired moment value
        """
        alpha, sigma, pi_0, mu, _ = self.para
        At = exp(-alpha*t)*pi_0 + mu*(1-exp(-alpha*t))
        Bt = sigma*exp(-alpha*t)
        res = 0
        for i in range((n+1)//2):
            res += comb(n, i)*At**(n-2*i)*Bt**(2*i) * \
                ((exp(2*alpha*t-1)-1)/(2*alpha))**(2*i)
        return res

    def moment(self, matrix: np.array, n: int, N: int)->float:
        """get empirical moment

        Args:
            matrix (np.array): simulated data
            n (int): order of moment
            N (int): number of simulation

        Returns:
            float: _description_
        """
        if n == 1:
            moment_array = np.sum(matrix, axis=0)
        else:
            moment_array = np.sum(np.power(matrix, n), axis=0)
        return moment_array/N

    def access_CIR(self, year: int, N: int, n: int, schemas: list[str])->None:
        """Access simulated CIR process

        Args:
            year (int): max year
            N (int): simulation times
            n (int): moment order
            schemas (list[str]): discretization method: QE, ZERO, ABS, INVERSE
        """

        CIRs = {s: [] for s in schemas}
        for i in tqdm(range(N)):
            seed = int(i+1)
            SRP = SimulationRiskPrime(self.para, seed)
            for schema in schemas:
                if schema in ['QE', 'INVERSE']:
                    dists = SRP.get_distrubution_sample("unif", year*settings.constant.DISCERT_DT_N)
                else:
                    dists = SRP.get_distrubution_sample("norm", year*settings.constant.DISCERT_DT_N)
                CIRs[schema].append(SRP.generate_CIR_process(schema, dists))

        res = {}
        for schema in schemas:
            CIRs[schema] = np.array(CIRs[schema])
            res[schema] = self.moment(CIRs[schema], n, N)

        theo_moments = [self.theo_CIR_moment(
            n, t/settings.constant.DISCERT_DT_N) for t in range(settings.constant.DISCERT_DT_N*year)]

        for schema in schemas:
            empi_m = res[schema]
            plt.plot(range(len(empi_m)), empi_m,
                     label=f"{schema} empirical moment {n}")
        plt.plot(range(len(theo_moments)), theo_moments,
                 label=f"theoretical moment {n}")
        plt.legend()
        plt.show()

