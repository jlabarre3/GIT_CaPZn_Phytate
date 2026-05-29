# **********************************************************************************************************************
# Julien Labarre, Christelle Loncke, Agnès Narcy, Maamer Jlali, Patrick Schlegel, Philippe Schmidely, 
# Marie-Pierre Létouneau-Montminy
# Modelling of calcium, phosphorus, and zinc fluxes in the gastrointestinal tract of growing pigs
# **********************************************************************************************************************

from function import differential_eq, feeding, calculate_fluxes, calculate_pH, name_pools, names_fluxes
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

# Define the scenario
filename = 'example_diet.csv'
path = 'diet/' + filename
df = pd.read_csv(path)
df = df.set_index(df.columns[0]) # use first column as index
diet0 = df.iloc[0].to_dict()
diet1 = df.iloc[1].to_dict()
diet2 = df.iloc[2].to_dict()
diet3 = df.iloc[3].to_dict()
diet4 = df.iloc[4].to_dict()

# import parameter 
params_solp = pd.read_csv('VF/parameters/phosphorus_bioavailability.csv', header=None,
                     dtype={0: str}, delimiter=';').set_index(0).squeeze().to_dict()
params = pd.read_csv('VF/parameters/model_parameters.csv', header=None,
                     dtype={0: str}, delimiter=';').set_index(0).squeeze().to_dict()

phytm_rel_act = pd.read_csv('VF/parameters/rel_quantumblue.txt', delimiter='\t', encoding= 'utf-8', decimal=',')

phytv_rel_act = pd.read_csv('VF/parameters/rel_phytv.txt', delimiter=';', encoding= 'utf-8', decimal=',')


dur = 1441
dt = 1
t = np.arange(0, dur, dt)

# create meal input vector 
feed_start = 0
feed_end = 720
t = np.arange(0, dur, dt)
meal_input = feeding(t, nmeals = 6, dmi=1.8, meal_dur=10, feed_start=feed_start, feed_end=feed_end, dt=dt)

scen = [diet0, diet1, diet2, diet3, diet4]
scen_name = ['LCa-PhytM0', 'LCa-PhytM500', 'CCa-PhytM0', 'CCa-PhytM500', 'CCa-PhytM2000']

j = 0
for diet in scen:
    print(j,'/',len(scen))
    print(diet)
    print(scen_name[j])
    nseg = 180
    P0S = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.162, 0.162, 0]) # Initial pools of the model, stomach
    PX = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Initial pools of the model, small intestine
    P0 =  np.repeat(PX[None, :], nseg, axis=0).reshape(14 * (nseg)) # Repeat initial pools for the number of segments
    y0 = np.concatenate((P0S, P0), axis=0)
    pHs_values = np.zeros(dur)
    model = np.zeros((dur, len(y0)))
    model[0, :] = y0

    for i in range(dur - 1):
    # compute model step
        sol = solve_ivp(differential_eq, [0, 1], model[i, :],method = 'RK45', args=(params, params_solp, meal_input[i], diet, dur, phytm_rel_act, phytv_rel_act, nseg))
        model[i + 1, :] = sol.y[:, -1]
    
    model = pd.DataFrame(data=model)
    print(model.shape)
    model = name_pools(model, nseg)
    model.insert(0, "Time", t, True)


    df_f = calculate_fluxes(model, params, params_solp, meal_input, diet, dur, phytm_rel_act, phytv_rel_act, nseg)
    df = calculate_pH(model)
    df_f = names_fluxes(df_f, nseg) # Name the fluxes
    df_t = pd.concat([df,df_f], axis = 1) # Concatenate pools and fluxes

    df_t.to_csv(f'example_{scen_name[j]}_may2026_v3.csv')

    print(df_t)
    j += 1
