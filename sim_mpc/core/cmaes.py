''' 
Lightweight CMA-ES implementation based on Nikolaus Hansen, 2003-09. 

For more information see Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772.

Authors: Markus Klar
Date: 11.2022
'''
import numpy as np
import math
import pandas as pd
from concurrent.futures import ProcessPoolExecutor as Pool
import concurrent.futures
import logging as log
import time
import numpy.matlib as matlib
import cfut


def worker_init(func):
    global _func
    _func = func

def worker(x):
    return _func(x)

def worker_cfut(x):
    return _job(x)

def xmap(func, iterable, processes):
    with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
        return(p.map(worker, iterable))

def boxarx(newx, boxconst):
    for k in range(newx.shape[0]):
        newx[k] = min(max(boxconst[k][0], newx[k]), boxconst[k][1])
    return newx

def cmaes(xstart, start_sigma, mainfolder, job, args, max_processes, max_iterations=-1, boxconst=None, verbosity=False, num_samples_costum=-1, use_cfut=False):
    # --------------------  Initialization --------------------------------  
    # User defined input parameters
    N = xstart.shape[0]                      # number of objective variables/problem dimension
    xmean = xstart.reshape(N,1)
    if verbosity:
        log.debug("cmaes_xstart = {}".format(xmean))

    log.debug(f"cmaes_N = {N}")
    
    sigma = start_sigma          # coordinate wise standard deviation (step size)
    stopfitness = 1e-4           # stop if fitness < stopfitness (minimization)


    # Strategy parameter setting: Selection  
    if num_samples_costum > 0:
        num_samples = num_samples_costum
    else:
        num_samples = 4+math.floor(3*math.log(N))                   # population size, offspring number
    
    mu = num_samples/2                                          # number of parents/points for recombination
    weights = np.resize(math.log(mu+1/2) - np.transpose(np.log(np.array(range(math.floor(mu)))+1)),(math.floor(mu),1))     # muXone array for weighted recombination     
    mu = math.floor(mu)
    weights = weights/sum(weights)                               # normalize recombination weights array
    mueff=sum(weights)**2/sum(weights**2)                        # variance-effectiveness of sum w_i x_i

    if max_iterations < 0:
        stopeval =  1e3*N**2            # stop after stopeval number of function evaluations
    else:
        stopeval = num_samples* max_iterations

    # Strategy parameter setting: Adaptation
    cc = (4+mueff/N) / (N+4 + 2*mueff/N)                     # time constant for cumulation for C
    cs = (mueff+2) / (N+mueff+5)                             # t-const for cumulation for sigma control
    c1 = 2 / ((N+1.3)**2+mueff)                               # learning rate for rank-one update of C
    cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)**2+mueff)) # and for rank-mu update
    damps = 1 + 2*max(0, math.sqrt((mueff-1)/(N+1))-1) + cs      # damping for sigma usually close to 1

    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros((N,1))                     # evolution paths for C and sigma
    ps = np.zeros((N,1))    
    B = np.eye(N)                       # B defines the coordinate system
    D = np.ones((N,))                      # diagonal D defines the scaling
    C = B @ np.diag(np.square(D)) @ B.T         # covariance matrix C
    invsqrtC = B @ np.diag(np.power(D,-1)) @ B.T   # C^-1/2 
    if verbosity:
        log.debug("INCV")
        log.debug(invsqrtC)
        log.debug(B * np.diag(np.power(D,-1)) * B.T)
    eigeneval = 0                      # track update of B and D
    chiN=N**0.5*(1-1/(4*N)+1/(21*N**2)) # expectation of ||N(0,I)|| == norm(randn(N,1)) 
    
    #Storage vars
    JJ = []
    XX = []
    SIGMA = []
    MEAN = []
    
    # Prepare output
    out_columns_x = [f"x_{k}_{i}" for k in range(num_samples) for i in range(N)]
    out_columns_fit = [f"fitness_{k}" for k in range(num_samples)]
    out_columns_sig = ["sigma"]
    out_columns_mean = [f"mean_{i}" for i in range(N)]
    
    log.debug("#################### Starting cma-es ############# \nnum_samples = {}, mu = {}, weights = {}".format(num_samples, mu, weights))
    #-------------------- Generation Loop --------------------------------
    counteval = 0  
    iteration = 0
    while counteval < stopeval:
        arx = np.zeros((N,num_samples))
        arfitness = np.zeros((num_samples,))

        # Generate and evaluate lambda offspring 
        for k in range(num_samples):
            rand=np.random.normal(size =(N,1))
            newx = np.reshape(xmean + sigma * B @ np.multiply(np.reshape(D,(N,1)), rand),(N,)) # m + sig * Normal(0,C) 
            
            if boxconst is not None:
                arx[:,k] = boxarx(newx, boxconst)
            else:
                arx[:,k] = newx

        # in parallel
        if use_cfut:
            start = time.time()

            def job_cfut(k, job, arx, args):
                return job(arx[:,k], counteval+k, args)


            with cfut.SlurmExecutor() as executor:  
                futures = []
                for k in range(num_samples):
                    global _job
                    _job = job_cfut
                    futures.append(executor.submit(worker_cfut, k, job=job, arx=arx, args=args) )
            for future in concurrent.futures.as_completed(futures):
                print(future.result())

            end = time.time()
        else:           
            start = time.time()
            arfitness = np.array(list(xmap(lambda k: job(arx[:,k], counteval+k, args), range(num_samples), max_processes)))
            end = time.time()
            
        counteval += num_samples
        
        arfitness.reshape((num_samples,))
        if verbosity:
            log.debug(f"{iteration}. run finished. Time for step: {end-start}s - Counteval: {counteval} - arx: {arx} - arfitness: {arfitness}")

        # Sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(arfitness)  # minimization
        arfitness = np.sort(arfitness)
        xold = xmean
        
        xmean = np.reshape(arx[:,arindex[0:mu]],(N,mu)) @ weights   # recombination, new mean value
        if verbosity:
            log.debug("xmean = {}".format(xmean))
        
        # Cumulation: Update evolution paths
        ps = (1-cs)*ps + math.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (xmean-xold) / sigma 
        
        hsig = np.sum(np.square(ps))/(1-(1-cs)**(2*counteval/num_samples))/N < 2 + 4/(N+1)
        
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma
        if verbosity:
            log.debug("ps = {}".format(ps))
            log.debug("hsig = {}".format(hsig))
            log.debug("oldC = {}".format(C))

        # Adapt covariance matrix C
        artmp = (1/sigma) * (np.reshape(arx[:,arindex[0:mu]],(N,mu))-matlib.repmat(xold,1,mu))
        C = (1-c1-cmu) * C + c1 * (pc @ pc.T + (1-hsig) * cc*(2-cc) * C) + cmu * artmp @ np.diag(np.resize(weights,(mu,))) @ artmp.T
        # regard old matrix, plus rank one update, minor correction if hsig==0, plus rank mu update
        if verbosity:
            log.debug("newC = {}".format(C))
            # Adapt step size sigma
            log.debug("sigmaold= {} - adaptval = {} - cs={} - damps={} - ps= {} - chiN={}".format(sigma, np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1)), cs,damps,ps,chiN))
        sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))

        if verbosity:
            log.debug("sigma={}".format(sigma))
        # Decomposition of C into B*diag(D.^2)*B' (diagonalization)
        if counteval - eigeneval > num_samples/(c1+cmu)/N/10:  # to achieve O(N^2)
            if verbosity:
                log.debug("updating B, D...")
            eigeneval = counteval

            C = np.triu(C) + np.triu(C,1).T     # enforce symmetry 
            [D,B] = np.linalg.eig(C)    # eigen decomposition, B==normalized eigenvectors
            if verbosity:
                log.debug("Eigenvalues = {}".format(D))
            D = np.sqrt(D)                            # D is a vector of standard deviations now
            invsqrtC = B @ np.diag(np.power(D,-1)) @ B.T
            if verbosity:
                log.debug("B = {} \n D = {} \n C = {}".format(B,D, C))
            
        # Save results
        JJ.append(arfitness)
        XX.append(arx[:,arindex].transpose().flatten())
        SIGMA.append(sigma)
        MEAN.append(xmean.flatten())
        
        if verbosity:
            log.debug(f"JJ = {JJ},\nXX = {XX},\nSIGMA = {SIGMA},\nMEAN = {MEAN}")
        dffit = pd.DataFrame(data=JJ, columns = out_columns_fit)
        dfx = pd.DataFrame(data=XX, columns = out_columns_x)
        dfs = pd.DataFrame(data=SIGMA, columns = out_columns_sig)
        dfm = pd.DataFrame(data=MEAN, columns = out_columns_mean)
        df = pd.concat([dffit, dfx, dfs, dfm], axis=1)

        df.to_csv(mainfolder + '/CMAES_result.csv', index=False)
        if verbosity:
            log.debug(f"Iteration {iteration} saved to {mainfolder + '/CMAES_result.csv'}.")

        # Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable 
        if arfitness[0] <= stopfitness or np.max(D) > 1e7 * np.min(D):
            terminalstring =""
            if arfitness[0] <= stopfitness:
                terminalstring = "Best fitness is lower than boundary. Terminating cma-es. arfitness[0]= {}, stopfitness= {}".format(arfitness[0],stopfitness)
                log.debug(terminalstring)
                
            else:
                terminalstring = "D Matrix condition exceeded limit. Terminating cma-es. np.max(D)= {}, np.min(D)= {}".format(np.max(D),np.min(D))
                log.debug(terminalstring)
            break
            
        

    # end while, end generation loop
    log.debug(f"Result: {arx[:, arindex[0]]}")
    return arx[:, arindex[0]], arfitness[arindex[0]] # Return best point of last iteration.