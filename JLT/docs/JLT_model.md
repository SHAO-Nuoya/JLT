# Model
## Basics
Risky zero coupon bond price with rating i:
\begin{equation}
    D_i(t,T)=B(t,T)(\delta + (1-\delta)\mathbb{Q}(\tau >T|\eta _t=i))
\end{equation}


Note $f_{i}(t, T)$ the risky forward rate and $f(t, T)$ the risk-free forward rate
\begin{equation}
    f_{i}(t, T)=f(t, T)+1_{\tau>t} \log \left(\frac{\delta+(1-\delta) \mathbb{Q}\left(\tau>T \mid \eta_{t}=i\right)}{\delta+(1-\delta) \mathbb{Q}\left(\tau>T+1 \mid \eta_{t}=i\right)}\right)
\end{equation}
Then the spread forward $s_{i}(t, T)$ can be defined as
\begin{equation}
s_{i}(t, T)=\log \left(\frac{\delta+(1-\delta) \mathbb{Q}\left(\tau>T \mid \eta_{t}=i\right)}{\delta+(1-\delta) \mathbb{Q}\left(\tau>T+1 \mid \eta_{t}=i\right)}\right)
\end{equation}

## Discrete case
### Transition matrix
Note $P^H$ the historical transition matrix, $P$ the risk-free transition matrix. And $p$ represent the probability from state i to state j in one year. Kijima and Komorobayashi (1999)\cite{Kijima97} and Lando (2000)\cite{lando2000some}. Jarrow, Lando and Turnbull propose
\begin{equation}
    p_{i,j}(t)=\pi_i(t)p_{i,j}^H,\ \forall i\neq j
\end{equation}
In matrix form
\begin{equation}
    P(t)=id+\Pi(t)[P^H-id]
\end{equation}
$\pi$ will be modeled by CIR process hence we need to initialize $\pi$ for each state $i$
\begin{equation}
    \pi_{i}(0)=\frac{B(0,1)-D_{i}(0,1)}{B(0,1)(1-\delta) p_{i, K}^{H}}
\end{equation}
Initialisation proof
\begin{aligned}
    B(0,1)-D_i(0,1)&=B(0,1)(1-\delta)(1-\mathbb{Q}(\tau > 1|\eta_0=i))\\
    &=B(0,1)(1-\delta)\mathbb{Q}(\tau < 1|\eta_0=i)\\
    &=B(0,1)(1-\delta)p_{i,K}\\
    &=B(0,1)(1-\delta)\pi_i p_{i,K}^H
\end{aligned}
Then by recurrence :
\begin{equation}
    \pi_{i}(t)=\sum_{j=1}^{K} p_{i, j}^{-1}(0, t) \frac{B(0, t+1)-D_{i}(0, t+1)}{(1-\delta) B(0, t+1) p_{i, K}^{H}}
\end{equation}

### Spread formula
The spread formula can be calculated by using
\begin{equation}
    s_i(0,t)=-\frac{ln\frac{D_i(0,t)}{B(0,t)}}{t}
\end{equation}
This formula comes from continuous model

## Continuous case
### Spread formula
We note $s_{i}^{f}(t, T)$ instantaneous spread forward and $s_{i}^{m}(t, T)$ the average spread forward which will be used for pricing
\begin{aligned}
    s_{i}^{f}(t, T)=\frac{(1-\delta) \frac{\partial p_{i K}(t, T)}{\partial T}}{1-(1-\delta) p_{i K}(t, T)}\\ s_{i}^{m}(t, T)=-\frac{\ln \left(\frac{D_{i}(t, T)}{B(t, T)}\right)}{T-t}=-\frac{\ln \left(1-(1-\delta) p_{i K}(t, T)\right)}{T-t} \label{10}
\end{aligned}
### Generator
In continuous case, risk prime $\Pi(t)$ will be defined by the generators 
\begin{gather}
    \Lambda^{H}=\Sigma D \Sigma^{-1} \\
    \Lambda(t)=\Sigma \Pi(t) D \Sigma^{-1}
\end{gather}
where $\Lambda^{H}$ is the historical generator and $\Lambda$ the risk-free one

### Transition matrix
Conditional transition $P^\pi(t,T)$ matrix and unconditional transition matrix $P^{\pi_t}(t,T)$ are defined as following:
\begin{equation}
    P^\pi(t,T) = e^{\Sigma\int_t^T\pi(s)dsD\Sigma ^{-1}}
\end{equation}
This transition matrix knowing the process $\pi$ from t to T is the matrix transition of an in-homogeneous Markov process
\begin{equation}
    P^{\pi_t}(t,T) = E^{\mathbb{Q}}[P^\pi(t,T)|F_t]
\end{equation}
These matrices are no longer the transition matrices of the process since they no longer respect the semi-group property $P(t, T) \neq P(t, s)P(s, T)$. However the sum of the rows is always 1 (by linearity of expectation). They are therefore always transition matrices.<br>
Note $MV(t)$ the obligation price, then we have
\begin{equation}
    MV(t)=\sum_{i=t}^{T} E^{\mathbb{Q}}\left[C F L_{i} \mid \mathcal{F}_{t}\right] B(t, i)\left(1-(1-\delta) P^{\pi_{t}}(t, i)_{\eta_{t}, \text { defaut }}\right)
\end{equation}
The closed formula allows to write the unconditional transition matrix as a deterministic function of $\pi_t$ ,we note $f$ this function.
$$P^{\pi_t}(t,T)=f(\pi_t, T-t, \beta)$$
Then by \eqref{10}, we know that the spread is function of unconditional transition matrix. Hence we have
\begin{equation}
    s_{\eta_{t}}(t, T)=g\left(P^{\pi_{t}}(t, T)\right)=g\left(f\left(\pi_{t}, T-t, \beta\right)\right)
\end{equation}
### Risk prime process : CIR
\begin{cases}
\pi_t=\alpha(\mu-\pi_t) d t+\sigma \sqrt{\pi_t} dW_{t} \\
\pi_{0}=\pi_{0}
\end{cases}

Then we can get the close formula for unconditional transition matrix

\begin{eqnarray}
    E^{\mathbb{Q}}\left[e^{d_{j} \int_{t}^{T} \pi(s) d s} \mid \mathcal{F}_{t}\right]&=&e^{A_{j}(t, T)-\pi(t) B_{j}(t, T)}\\
    A_{j}(t, T)&=&\frac{2\alpha\mu}{\sigma^{2}}\ln\left(\frac{2\nu_{j}e^{\frac{\left(\alpha+\nu_{j}\right)(T-t)}{2}}}{\left(\alpha+\nu_{j}\right)\left(e^{\nu_{j}(T-t)}-1\right)+2 \nu_{j}}\right) \\
    B_{j}(t, T)&=&-\frac{2d_{j}\left(e^{\nu_{j}(T-t)}-1\right)}{\left(\alpha+\nu_{j}\right)\left(e^{\nu_{j}(T-t)}-1\right)+2 \nu_{j}} \\
    \nu_{j}&=&\sqrt{\alpha^{2}-2 d_{j} \sigma^{2}}
\end{eqnarray}

The simulation of CIR process is based on \cite{BGAndersen}
+ Given $\pi_t$, compute $m$ and $s^2$ 
+ Compute $\psi = s^2/m^2$ 
+ Draw a uniform random number $U_V$ 

+ If $\psi < \psi_c$ : 
    * Compute $a$ and $b$
    * Compute $Z_V = \Phi^{-1} (U_V )$
    * set $\pi_{t+\Delta} = a(b+Z_V)^2$

+ Otherwise, $\psi > \psi_c$ :
    * Compute $\beta$ and $p$
    * set $\pi_{t+\Delta} = \Psi^{-1}(U_V, p, \beta)$

where 
\begin{align}
    m &= \mu+(\pi_t-\mu)e^{-\alpha\Delta} \\
    s^{2}&=\frac{\pi_t \sigma^{2}e^{-\alpha\Delta}}{\alpha}\left(1-e^{-\alpha\Delta}\right)+\frac{\mu \sigma^{2}}{2 \alpha}\left(1-e^{-\alpha \Delta}\right)^{2} \\
    \psi&=\frac{s^{2}}{m^{2}}=\frac{\frac{\pi_t\sigma^{2}e^{-\alpha\Delta}}{\alpha}\left(1-e^{-\alpha \Delta}\right)+\frac{\mu\sigma^{2}}{2\alpha}\left(1-e^{-\alpha\Delta}\right)^{2}}{\left(\mu+(\pi_t-\mu)e^{-\alpha \Delta}\right)^{2}} \\
    \psi_c&=1.5 \\
    b^{2}&=2 \psi^{-1}-1+\sqrt{2 \psi^{-1}} \sqrt{2 \psi^{-1}-1} \\
    a&=\frac{m}{1+b^{2}} \\
    p&=\frac{\psi-1}{\psi+1} \\
    \beta&=\frac{1-p}{m}=\frac{2}{m(\psi+1)} \\
    \Psi^{-1}(u)&=\Psi^{-1}(u ; p, \beta)= 
    \begin{cases}
        0,  0 \leq u \leq p \\ 
        \beta^{-1} \ln \left(\frac{1-p}{1-u}\right),  p<u \leq 1
    \end{cases}
\end{align}

### Default probability
\begin{gather*}
    \begin{aligned}
        &P^{\pi_{t}}(t, T)&=&\Sigma \operatorname{diag}\left(E\left[e^{\int_{t}^{T} d_1\Pi(s) d s}\right], \ldots\right) \Sigma^{-1} \\
        &P^{\pi_{t}}(t, T)-I d&=&\Sigma\left(\operatorname{diag}\left(E\left[e^{\int_{t}^{T} d_1\Pi(s) d s}\right], \ldots\right)-I d\right) \Sigma^{-1}
    \end{aligned}
\end{gather*}
Finally, the default probability can be written as
\begin{equation}
\begin{aligned}
        p_{i, K}(t, T)&=(P^{\pi_{t}}(t, T)-I d)_{i, K}\\
        &=\sum_{j=1}^{K-1} \sigma_{i, j}\left(\sigma^{-1}\right)_{j, K}\left(e^{A_{j}(t, T)-\pi(t) B_{j}(t, T)}-1\right)
    \end{aligned}
\end{equation}




# Calibration
## Transition matrix
\begin{itemize}
    \item Historical matrix : In the article, obtained from Moody's study
    \item Risk neutral matrix : Risk prime multiply historical one
\end{itemize}
##Recovery rate
In the article, obtained from Moody's study
## Spread

## Calibration process
+ Historical calibration : Given moments of spread, based on moments methods, used for initialisation of parameters under risk neutral measure.
+ Calibration of $\pi_0$ : can be calibrated with historical transition matrix and market spread by using the formula

\begin{equation}
    \frac{p_{i,k}(0,T)}{T}=\lambda_{iK}^{RN}=\lambda_{iK}^{H}\pi_0 \label{19}
\end{equation}
where $p_{i,k}(0,T)$ is the probability of default
+ Least square: minimize loss function
\begin{equation}
\begin{aligned}
    f(\alpha, \mu, \sigma, \pi _0)&=\sum_{t=1}^{Maturity}\sum_{i=1}^{State\ number}(s^{model}_i(0,t)-s_i^{market}(0,t°))^2\\
    &+1000000*(\sigma -init\sigma)\\
    &+1000000*(\alpha -init\alpha)\\
    &+10000*(\pi _0 -\mu)\\
    &+10000*(\pi _0 -init\pi _0)\\
\end{aligned}
\end{equation}
