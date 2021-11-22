import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

from tqdm import tqdm
#Agent based
_nAgents = 200000
state = np.zeros(_nAgents)
data = {"state": state}
df = pd.DataFrame(data)
df.describe()

def infect(df, contacts, probability=1.0):
    unique, counts = np.unique(contacts, return_counts=True)

    roll = np.random.uniform(0,1,len(unique))

    state = df.loc[unique,"state"]
    probability = 1 - np.power(1-probability, counts)
    change = np.array(roll <= probability).astype(int)

    df.loc[unique,"state"] = state + np.maximum(change*(1-state),0)


def init(nAgents=1000, nPatientZero=1):
    state = np.zeros(nAgents)

    neighborhood = np.zeros(nAgents)
    data = {"state": state, "neighborhood": neighborhood}

    df = pd.DataFrame(data)
    patientZero = np.random.choice(df.index, nPatientZero, replace=False)
    infect(df, patientZero, probability=1.0)
    return df

def recover(df, probability=1.0):
    roll = np.random.uniform(0,1,len(df[df["state"] == 1]))
    chance = np.array(roll <= probability).astype(int)

    df.loc[df["state"] == 1,"state"] = 1 + chance

def step(df):
    nInfected = np.sum(df["state"] == 1)
    contacts = np.random.choice(df.index, _randomContacts * nInfected, replace=True)

    infect(df, contacts, _chanceOfInfection)
    recover(df, _chanceOfRecovery)

def simulate(df, stats, nSteps=100, mode="random", nRandomContacts=0, plotLattice=False):
    for i in tqdm(range(nSteps)):
        step(df)

        stats["nSusceptible"].append(np.sum(df["state"] == 0))
        stats["nInfected"].append(np.sum(df["state"] == 1))
        stats["nRemoved"].append(np.sum(df["state"] == 2))

_randomContacts = 5 #modifiqué de 9 a 5
_chanceOfInfection = 0.01 #modifiqué de 0.025 a 0.05
_daysCuredAfter = 10 #cambié numero de día
_chanceOfRecovery = 1./_daysCuredAfter

_nExperiments = 10
_nAgents = 200000
_nSteps = 100

_nPatientZero = 50000

x = np.linspace(0,_nSteps-1,_nSteps)
allStats = []

#SIRModel = MathModel(_nAgents, _nPatientZero, chanceOfInfection=0.025, randomContacts=5, daysCuredAfter=21)
#print("Simulate SIR Model")
#for i in tqdm(range(_nSteps)):
 #   SIRModel.calculateStep()

for iExp in range(_nExperiments):
    print("Starting Experiment:",iExp+1,"/",_nExperiments)
    st = {"nInfected": [], "nRemoved": [], "nSusceptible": []}

    df = init(_nAgents, _nPatientZero)

    simulate(df, stats=st, nSteps=_nSteps)

    allStats.append(st)

def calculateStats(allStats):
    medianStats = dict()
    lowerStats = dict()
    higherStats = dict()

    for key in allStats[0]:
        l = []
        for st in allStats:
            l.append(st[key])
        a = np.stack(l)
        medianStats[key] = np.median(a, axis=0)
        lowerStats[key] = np.quantile(a, 0.25, axis=0)
        higherStats[key] = np.quantile(a, 0.75, axis=0)

    return medianStats, lowerStats, higherStats

def plotSIR(x,mdianStats,lowerStats,higherStats,figName="tmp.png"):
    np.savetxt("data.txt",medianStats["nInfected"],fmt='%1e',delimiter=",")
    plt.plot(x,medianStats["nSusceptible"], color = "green", label="Susceptible")
    plt.plot(x,medianStats["nInfected"], color="red", label="Infected")
    plt.plot(x,medianStats["nRemoved"], color="blue", label="Recovered")
    #plt.plot(x,nDead, color="black")
    plt.fill_between(x, lowerStats["nSusceptible"], higherStats["nSusceptible"],
                     color='green', alpha=0.1)
    plt.fill_between(x, lowerStats["nInfected"], higherStats["nInfected"],
                     color='red', alpha=0.1)
    plt.fill_between(x, lowerStats["nRemoved"], higherStats["nRemoved"],
                     color='blue', alpha=0.1)


    plt.xlabel("Time steps [days]")
    plt.ylabel("Number of cases")

    lgd = plt.legend(bbox_to_anchor=(1.01,0.65), loc="center left")
    plt.tight_layout()

    plt.savefig(figName, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()

medianStats, lowerStats, higherStats = calculateStats(allStats)
plotSIR(x,medianStats,lowerStats,higherStats,figName="CompareToSIR.png")
