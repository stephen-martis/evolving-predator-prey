import numpy as np
import math
import pickle as pickle
import copy
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--mutation_prob', default=0.1, type=float)
parser.add_argument('-v', '--var', default=0.1, type=float)
parser.add_argument('-L', '--genome_length', default=6, type=float)
parser.add_argument('-n', '--label', default='test', type=str)
args = parser.parse_args()


def int_to_bin(n, L):
    a = [int(s) for s in "{0:b}".format(n).zfill(L)]
    return a

def bin_to_int(a):
    n = 0
    for i in range(len(a)):
        n+=a[i]*2**i
    return n

def derivative(y, growth_rates, interactions, dep_graph):
    vec = np.dot(growth_rates, y)+np.dot(dep_graph, np.multiply(y, np.dot(interactions, y)))
    return vec

def RK4_burn_in(u0, tf, g, A, h, L, p):
    
    print('Creating interactions...')
    hypercube = np.zeros((2**L, 2**L))
    np.fill_diagonal(hypercube, 1-p)
    for i in range(2**L):
        for j in range(L):
            temp = int_to_bin(i, L)
            temp[j]=1-temp[j]
            hypercube[i][bin_to_int(temp)] = p/L  
        
    dep_graph = np.block([[np.identity(2**L), np.zeros((2**L, 2**L))],
                          [np.zeros((2**L, 2**L)), hypercube]])
    
    T = np.arange(0, tf, h)
    ub = u0
    u = []
    print("Simulating...")
    
    count = 0
    for i in np.arange(len(T)):

        k1 = derivative(ub, g, A, dep_graph)
        y1 = ub+k1*h/2.0

        k2 = derivative(y1, g, A, dep_graph)
        y2 = ub+k2*h/2.0

        k3 = derivative(y2, g, A, dep_graph)
        y3 = ub+k3*h

        k4 = derivative(y3, g, A, dep_graph)
        
        #with extinction threshold
        #ub = (ub + (k1+2*k2+2*k3+k4)*h/6.0)*np.heaviside(ub + (k1+2*k2+2*k3+k4)*h/6.0 - thresh, 0.0)
        ub = (ub + (k1+2*k2+2*k3+k4)*h/6.0)
        
        if count%500==0:
            u.append(copy.deepcopy(ub))
        count+=1
    return u

L = int(args.genome_length)
mean = 1e-3
var = float(args.var)
sigma_sq = np.log(var/(mean**2)+1)
mu = np.log(mean)-sigma_sq/2.0
p = float(args.mutation_prob)

print('Creating interactions...')
prey_growth = np.zeros((2**L, 2**L))
np.fill_diagonal(prey_growth, 1.0-p)
for i in range(2**L):
    for j in range(L):
        temp = int_to_bin(i, L)
        temp[j]=1-temp[j]
        prey_growth[i][bin_to_int(temp)] = p/float(L)

pred_death = np.zeros((2**L, 2**L))
np.fill_diagonal(pred_death, -1.0)

growth_rates = np.block([
    [prey_growth, np.zeros((2**L, 2**L))],
    [np.zeros((2**L, 2**L)), pred_death] ])

a = np.random.lognormal(mean=mu, sigma=np.sqrt(sigma_sq), size=(2**L, 2**L))
s = np.random.lognormal(mean=mu, sigma=np.sqrt(sigma_sq), size=(2**L, 2**L))
A = np.block([
    [np.zeros((2**L, 2**L)), -a-s],
    [a, np.zeros((2**L, 2**L))] ])

u0 = 500.0/float(2**L)*np.ones(2**(L+1))
u0[:2**L] = 2.0*u0[:2**L]
#u0 = np.zeros(2**(L+1))
#u0[0] = 1000
#u0[2**L] = 500

h = 0.00001
tf = 80.0

t0 = time.time()
print('Calculating trajectories...')
sol = RK4_burn_in(u0, tf, growth_rates, A, h, L, p)
t1 = time.time()

print('Done!')
print('Time: '+str(t1-t0))

pickle.dump( [[L, mean, var], [h, tf], sol], open( "./deterministic_data/mutprob_var"+str(args.var)+"_p0.1_genome"+str(args.genome_length)+'_'+args.label+".p", "wb" ) )
