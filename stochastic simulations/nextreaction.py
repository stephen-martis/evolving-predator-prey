import numpy as np
import math
import random
from pqdict import PQDict
import copy
import time
import pickle as pickle
import argparse

#input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--mutation_prob', default=0.1, type=float)
parser.add_argument('-m', '--mu', default=0.1, type=float)
parser.add_argument('-s', '--sigma_sq', default=0.2, type=float)
parser.add_argument('-l', '--label', default='', type=str)
parser.add_argument('-f','--folder', default='', type=str)
args = parser.parse_args()

# helpers for mutation process
def bin_to_int(a):
    n = 0
    for i in range(len(a)):
        n+=a[i]*2**i
    return n

def int_to_bin(n, L):
    a = [int(s) for s in "{0:b}".format(n).zfill(L)]
    return a
    
def list_mutants(n, L):
    a = int_to_bin(n, L)
    mutants = []
    for i in range(len(a)):
        temp = a
        temp[i]=1-temp[i]
        mutants.append(bin_to_int(temp))
    return np.array(mutants)

# declare different reactions
class Birth(object):
    def __init__(self, mother, rate, L):
        self.mother = mother
        self.daughter = mother
        self.stoich = {self.daughter: 1}
        self.mutants = list_mutants(mother, L)
        self.rate_const = rate
    
    def offspring(self, mu):
        if np.random.choice([False, True], p=[1-mu, mu]):
            self.daughter = random.choice(self.mutants)
            self.stoich = {self.daughter: 1}
        else:
            self.daughter = self.mother
            self.stoich = {self.daughter: 1}
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]
    
class Death(object):
    def __init__(self, mother, rate, L):
        self.mother = mother
        self.stoich = {self.mother: -1}
        self.rate_const = rate
        
    def offspring(self, mu):
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]
        
class Predation(object):
    def __init__(self, mother, food, rate, L, daughter=None):
        self.mother = mother
        self.food = food
        self.stoich = {self.food: -1}
        self.rate_const = rate
        
    def offspring(self, mu):
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]*state[self.food]
    
class Predation_Birth(object):
    def __init__(self, mother, food, rate, L_prey, L_pred):
        self.mother = mother
        self.daughter = mother
        self.food = food
        self.stoich = {self.food: -1, self.daughter: 1}
        self.mutants = list_mutants(mother-2**L_prey, L_pred)+2**L_prey
        self.rate_const = rate
    
    def offspring(self, mu):
        if random.random() < mu:
            self.daughter = random.choice(self.mutants)
            self.stoich = {self.food: -1, self.daughter: 1}
        else:
            self.daughter = self.mother
            self.stoich = {self.food: -1, self.daughter: 1}

    def propensity(self, state):
        return self.rate_const*state[self.mother]*state[self.food]

       
#### Main loop ####

def main_simulation(reactions, inv_map, x, num_events, u):
# Generate random numbers, times, priority queue
    mu=u
    t=0
    T=[t]
    X=[copy.deepcopy(x)]
    schedule = PQDict()
    
    for i in range(len(reactions)):
        tau = -math.log(random.random())/reactions[i].propensity(x)
        schedule[i] = tau
    
    rnext, tnext = schedule.topitem()
    t = tnext
    dep_set = set()
    reactions[rnext].offspring(mu)
    for k, v in reactions[rnext].stoich.items():
        x[k]=x[k]+v
        dep_set.update(inv_map[k])
    n=1

    while n < num_events:
        # reschedule reaction
        tau = -math.log(random.random())/reactions[rnext].propensity(x)
        schedule[rnext] = t + tau

        # reschedule dependent reactions
        for r in dep_set:
            tau = -math.log(random.random())/reactions[r].propensity(x)
            schedule[r] = t + tau
        
        # next reaction
        rnext, tnext = schedule.topitem()
        t = tnext
        dep_set = set()
        reactions[rnext].offspring(mu)
        for k, v in reactions[rnext].stoich.items():
            x[k]=x[k]+v
            dep_set.update(inv_map[k])
        n+=1
        # choose when to record data
        if n%500==0:
            print(n)
            T.append(t)
            X.append(copy.deepcopy(x))

    return T, X



##### Initialize #####
print('initializing...')

# time
t = 0
tf = 10

# number of predator and prey sites
L_prey = 10
L_pred = 10

# number of predator and prey genotypes
N_prey = 2**L_prey
N_pred = 2**L_pred

##### Interaction helper functions #####

# convert raw mean and variance to lognormal parameters
int_var = np.log(args.sigma_sq/(args.mu**2)+1)
int_mean = np.log(args.mu)-int_var/2.0

##### Create reaction dictionaries #####

reactions = {}
rates = {}

# Births
prey_growth_mean = 1.0
prey_growth_scale = 0
bi = []
for i in range(N_prey):
    r = np.random.normal(loc=prey_growth_mean, scale=prey_growth_scale)
    reactions[i] = Birth(i, r, L_prey)
    bi.append(r)

rates['birth'] = bi
print('created birth reactions...')
    
# Deaths
pred_death_mean = 1.0
pred_death_scale = 0
di = []
for i in range(N_prey, N_prey+N_pred):
    r = np.random.normal(loc=pred_death_mean, scale=pred_death_scale)
    reactions[i] = Death(i, r, L_pred)
    di.append(r)

rates['death'] = di
print('created death reactions...')

# Predation/Birth
predation_birth_mean = int_mean
predation_birth_scale = np.sqrt(int_var)
aij = []
temp = 0
for i in range(N_prey, N_prey+N_pred):
    temp2 = []
    for j in range(N_prey):
        if i==N_prey and j==0:
            r = args.mu
            reactions[N_prey+N_pred+N_prey*N_pred+temp] = Predation_Birth(i, j, r, L_prey, L_pred)
            temp2.append(r)            
        else:
            r = np.random.lognormal(mean=predation_birth_mean, sigma=predation_birth_scale)
            reactions[N_prey+N_pred+N_prey*N_pred+temp] = Predation_Birth(i, j, r, L_prey, L_pred)
            temp2.append(r)
        temp+=1
    aij.append(temp2)
rates['predation birth'] = aij
print('created predation/birth reactions...')

# Predation
predation_mean = int_mean
predation_scale = np.sqrt(int_var)
sij = []
temp = 0
for i in range(N_prey, N_prey+N_pred):
    temp2 = []
    for j in range(N_prey):
        if i==N_prey and j==0:
            r = args.mu
            reactions[N_prey+N_pred+temp] = Predation(i, j, r, L_pred)
            temp2.append(r)
        else:    
            r = np.random.lognormal(mean=predation_mean, sigma=predation_scale)
            reactions[N_prey+N_pred+temp] = Predation(i, j, r, L_pred)
            temp2.append(r)
        temp+=1
    sij.append(temp2)
rates['predation'] = sij
print('created predation reactions...Done!')


##### Create reaction dependency graph #####
prod_dict = {}
react_dict = {}

for i in range(N_prey):
    prod_dict[i] = [reactions[i].mother]
    prod_dict[i].extend(reactions[i].mutants)
    react_dict[i] = [reactions[i].mother]
for i in range(N_prey, N_prey+N_pred):
    prod_dict[i] = [reactions[i].mother]
    react_dict[i] = [reactions[i].mother]
for i in range(N_prey+N_pred, N_prey+N_pred+N_prey*N_pred):
    prod_dict[i] = [reactions[i].mother]
    prod_dict[i].append(reactions[i].food)
    react_dict[i] = [reactions[i].mother, reactions[i].food]
for i in range(N_prey+N_pred+N_prey*N_pred, N_prey+N_pred+2*N_prey*N_pred):
    prod_dict[i] = [reactions[i].mother]
    prod_dict[i].extend(reactions[i].mutants)
    prod_dict[i].append(reactions[i].food)
    react_dict[i] = [reactions[i].mother, reactions[i].food]

inv_map = {}
for k, v in react_dict.items():
    for i in v:
        inv_map.setdefault(i,set()).add(k)

print('created dependency graph...Done!')

##### Initialize #####
x = np.zeros(N_prey+N_pred)
x[0] = 1000
x[N_prey] = 500

# number of interactions
num_events = 2000000

# time the simulation
t1 = time.time()

# run simulation
T,X = main_simulation(reactions, inv_map, x, num_events, args.mutation_prob)

# print simulation time
print('total time: '+str(time.time()-t1))

# save output
pickle.dump([T,X,reactions,inv_map,rates], open( "./data/mutprob_u"+str(args.mutation_prob)+"_mu"+str(args.mu)+"_var"+str(args.sigma_sq)+"_"+str(args.label)+".p", "wb" ) )
