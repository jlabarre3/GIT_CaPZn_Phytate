# This script produces Figure of the following manuscript: 
# Title: A mechansitic model of the digestive fluxes of calcium, phosphorus and zinc in growing pigs
# Authors: Julien Labarre, Philippe Scmidely, Agnès Narcy, Maamer Jlali, Christelle Loncke, Marie-Pierre Létourneau-Montminy
# *Corresponding author:
#           marie-pierre.letourneau-montminy.1@ulaval.ca
# 
# Out: Figure3.pdf
# Date: January 15, 2026
# Author: Julien Labbare <julien.labarre.1@ulaval.ca>


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from function import create_df_seg, sum_col_un

scen1 = pd.read_csv('/Users/julienlabarre/Documents/PhD/4-in_silico/VersionMay2026/example_LCa-PhytM0_may2026_v3.csv', sep = ',')
scen2 = pd.read_csv('/Users/julienlabarre/Documents/PhD/4-in_silico/VersionMay2026/example_LCa-PhytM500_may2026_v3.csv', sep = ',')
scen3 = pd.read_csv('/Users/julienlabarre/Documents/PhD/4-in_silico/VersionMay2026/example_CCa-PhytM0_may2026_v3.csv', sep = ',')
scen4 = pd.read_csv('/Users/julienlabarre/Documents/PhD/4-in_silico/VersionMay2026/example_CCa-PhytM500_may2026_v3.csv', sep = ',')
scen5 = pd.read_csv('/Users/julienlabarre/Documents/PhD/4-in_silico/VersionMay2026/example_CCa-PhytM2000_may2026_v3.csv', sep = ',')


scen = [scen1, scen2, scen3, scen4, scen5]
no_sim = len(scen)
nseg = 180
nmeals = 2
dmi = 2
sim_duration = 1441  
vars_to_save = ['SDPPs', 'SDPPi', 'SDNPPs', 'SDPPs:SDNPPs_phytm']
labels = ['LCa-PhytM0', 'LCa-PhytM500','CCa-PhytM0', 'CCa-PhytM500', 'CCa-PhytM2000']

''' plot SDPPs, SDPPi, SDNPPs and SDPPs:SDNPPs_phytm curves '''
t = np.arange(sim_duration)
c = plt.cm.ocean(np.linspace(0, 0.85, no_sim))
ltype = ['solid', 'dashed', 'solid', 'dashed', 'dotted']
xtick_pos = np.arange(0, 1441, 250)
ytick = [0, 1, 2, 3]

fig, ax = plt.subplots(2, 2, figsize=(7, 4.3), sharex=True)

for i in range(no_sim):
    df = scen[i]  # <-- le DataFrame du scénario i

    ax[0, 0].plot(t, df['SDPPi'].to_numpy(), color=c[i], linewidth=1.5, linestyle = ltype[i], label=labels[i])
    ax[0, 1].plot(t, df['SDPPs'].to_numpy(), color=c[i], linewidth=1.5, linestyle  = ltype[i], label=labels[i])
    ax[1, 0].plot(t, df['SDNPPs'].to_numpy(), color=c[i], linewidth=1.5, linestyle  = ltype[i], label=labels[i])
    ax[1, 1].plot(t, (df['SDPPs:SDNPPs_phytm'] * 1000).to_numpy(), color=c[i], linestyle  = ltype[i], linewidth=1.5, label=labels[i])


ax[0, 0].set_ylabel('SDPPi, g')
ax[0, 0].set_ylim(0, .8)
ax[0, 0].set_yticks([0, .15, .30, .45, .60, .75])
ax[0, 0].set_yticklabels([str(i) for i in [0, .15, .30, .45, .60, .75]])
ax[0, 0].grid()

ax[0, 1].set_ylabel('SDPPs, g')
ax[0, 1].set_ylim(0, .8)
ax[0, 1].set_yticks([0, .15, .30, .45, .60, .75])
ax[0, 1].set_yticklabels([str(i) for i in [0, .15, .30, .45, .60, .75]])
ax[0, 1].grid()

ax[1, 0].set_ylabel('SDNPPs, g')
ax[1, 0].set_xticks(xtick_pos)
ax[1, 0].set_xticklabels(xtick_pos)
ax[1, 0].set_xlim(0, 1440)
ax[1, 0].set_yticks([0, .25, .50, .75, 1.00, 1.25])
ax[1, 0].set_yticklabels([str(i) for i in [0, .25, .50, .75, 1.00, 1.25]])
ax[1, 0].set_xlabel('Temps, min')
ax[1, 0].grid()

ax[1, 1].set_ylabel('Hydrolyse PP, mg/min')
ax[1, 1].set_xticks(xtick_pos)
ax[1, 1].set_xticklabels(xtick_pos)
ax[1, 1].set_xlim(0, 1440)
ax[1, 1].set_yticks([0, 5, 10, 15, 20, 25])
ax[1, 1].set_yticklabels([str(i) for i in [0, 5, 10, 15, 20, 25]])
ax[1, 1].set_xlabel('Temps, min')
ax[1, 1].grid()

handles, labels_leg = ax[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels_leg,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8,
    handlelength=2,
    columnspacing=1.2,
    alignment="center"
)


fig.subplots_adjust(bottom=0.22)
for ax, lab in zip(ax.ravel(), "ABCD"):
    ax.text(0.90, 0.92, lab, transform=ax.transAxes,
            ha="left", va="top", fontweight="light", fontsize=14)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure5.png', dpi= 300,  bbox_inches="tight")
plt.show()

''' plot the different curves '''

t = np.arange(nseg-1)
c = plt.cm.ocean(np.linspace(0, 0.85, no_sim))
xtick_pos = np.arange(0, 190, 30)
ytick = [0, 20, 40, 60, 80, 100]

fig, ax = plt.subplots(2, 2, figsize=(7, 4.3))

for i in range(no_sim):
    df = scen[i]  # <-- le DataFrame du scénario i

    # compute intakes
    PPint = sum(df['EXT:SDPPi'])
    Pint = sum(df['EXT:SDNPPs']) + sum(df['EXT:SDNPPi'])+ sum(df['EXT:SDPPi'])
    Caint = sum(df['EXT:SDCaCO3'])+ sum(df['EXT:SDCas']) + sum(df['EXT:SDCav']) 
    Znint = sum(df['EXT:SDZns']) + sum(df['EXT:SDZnv'])
    
    # compute flux per seg
    df_seg1 = create_df_seg(df, df, nseg, flux='IDPPi')
    df_seg2 = create_df_seg(df, df, nseg, flux='IDPPs')
    df_seg3 = create_df_seg(df, df, nseg, flux='IDNPPi')
    df_seg4 = create_df_seg(df, df, nseg, flux='IDNPPs')
    df_seg5 = create_df_seg(df, df, nseg, flux='IDCai')
    df_seg6 = create_df_seg(df, df, nseg, flux='IDCas')
    df_seg7 = create_df_seg(df, df, nseg, flux='IDZni')
    df_seg8 = create_df_seg(df, df, nseg, flux='IDZns')

    # divide per P intake 
    df_seg1['IDPPi%'] = ((df_seg1['IDPPi']) / Pint) * 100
    df_seg2['IDPPs%'] = ((df_seg2['IDPPs']) / Pint) * 100
    df_seg3['IDNPPi%'] = ((df_seg3['IDNPPi']) / Pint) * 100
    df_seg4['IDNPPs%'] = ((df_seg4['IDNPPs']) / Pint) * 100
    df_seg5['IDCai%'] = ((df_seg5['IDCai']) / Caint) * 100
    df_seg6['IDCas%'] = ((df_seg6['IDCas']) / Caint) * 100
    df_seg7['IDZni%'] = ((df_seg7['IDZni']) / Znint) * 100
    df_seg8['IDZns%'] = ((df_seg8['IDZns']) / Znint) * 100

    # plot
    #ax[0, 0].plot(t, df_seg1['IDPPi%'], color=c[i], linewidth=1.5, linestyle='--', label=f'IDPPi-{labels[i]}')
    ax[0, 0].plot(t, df_seg2['IDPPs%'], color=c[i], linewidth=1.5, linestyle  = ltype[i], label=f'{labels[i]}')
    #ax[0, 1].plot(t, df_seg3['IDNPPi%'], color=c[i], linewidth=1.5, linestyle='--', label=f'IDNPPi-{labels[i]}')
    ax[0, 1].plot(t, df_seg4['IDNPPs%'], color=c[i], linewidth=1.5, linestyle  = ltype[i], label=f'{labels[i]}')
    #ax[1, 0].plot(t, df_seg5['IDCai%'], color=c[i], linewidth=1.5, linestyle='--', label=f'IDCai-{labels[i]}')
    ax[1, 0].plot(t, df_seg6['IDCas%'], color=c[i], linewidth=1.5, linestyle  = ltype[i], label=f'{labels[i]}')
    #ax[1, 1].plot(t, df_seg7['IDZni%'], color=c[i], linewidth=1.5, linestyle='--', label=f'IDZni-{labels[i]}')
    ax[1, 1].plot(t, df_seg8['IDZns%'], color=c[i], linewidth=1.5, linestyle  = ltype[i], label=f'{labels[i]}')

ax[0, 0].set_ylabel('IDPPs, % of intake')
ax[0, 1].set_ylabel('IDNPPs, % of intake')
ax[1, 0].set_ylabel('IDCas, % of intake')
ax[1, 1].set_ylabel('IDZns, % of intake')
ax[0, 0].set_ylim(00, 100)
ax[0, 0].set_yticks(ytick)
ax[0, 0].set_yticklabels([str(i) for i in ytick])
ax[0, 1].set_yticks(ytick)
ax[0, 1].set_yticklabels([str(i) for i in ytick])
ax[1, 0].set_yticks(ytick)
ax[1, 0].set_yticklabels([str(i) for i in ytick])
ax[1, 1].set_yticks(ytick)
ax[1, 1].set_yticklabels([str(i) for i in ytick])
ax[0, 0].grid()
ax[0, 1].grid()

#ax[1, 0].set_ylabel('Quantitées, % des apports')
ax[1, 0].grid()

ax[1, 1].grid()
ax[1, 1].set_xlabel('Segment')
ax[1, 0].set_xlabel('Segment')


handles, labels_leg = ax[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels_leg,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8,
    handlelength=2,
    columnspacing=1.2,
    alignment="center"
)

fig.subplots_adjust(bottom=0.22)
for ax, lab in zip(ax.ravel(), "ABCD"):
    ax.text(0.90, 0.92, lab, transform=ax.transAxes,
            ha="left", va="top", fontweight="light", fontsize=14)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure6.png', dpi= 300,  bbox_inches="tight")
plt.show()

out = {
    'phytm' : [],
    'aid_pp' : [],
    'attd_p' : [],
    'attd_ca' : [],
    'attd_zn' : []
}

''' plot response curve'''

for i in range(no_sim):
    df = scen[i]  # <-- le DataFrame du scénario i

    # compute intakes
    phytm = sum(df['EXT:SDPHYTM'])
    Pint = sum(df['EXT:SDNPPs']) + sum(df['EXT:SDNPPi'])+ sum(df['EXT:SDPPi'])
    PPint = sum(df['EXT:SDPPi'])
    Caint = sum(df['EXT:SDCaCO3'])+ sum(df['EXT:SDCas']) + sum(df['EXT:SDCav']) 
    Znint = sum(df['EXT:SDZns']) + sum(df['EXT:SDZnv'])

    unpabs = sum_col_un(df, ['IDNPPs', 'IDNPPi', 'IDPPs', 'IDPPi', 'IEPs', 'IEPi'], nseg)
    unppabs = sum_col_un(df, ['IDPPs', 'IDPPi'], nseg)
    uncaabs = sum_col_un(df, ['IDCas', 'IDCai', 'IECas', 'IECai'], nseg)
    unznabs = sum_col_un(df, ['IDZns', 'IDZni', 'IEZns', 'IEZni'], nseg)

    aid_pp = (PPint - unppabs)/PPint * 100
    attd_p = (Pint - unpabs)/Pint * 100
    attd_ca = (Caint - uncaabs)/Caint * 100
    attd_zn = (Znint - unznabs)/Znint * 100

    print(i)
    print(phytm)
    print(PPint)
    print(Pint)
    print(Caint)
    print(Znint)
    print(aid_pp)
    print(attd_p)
    print(attd_ca)
    print(attd_zn)

    out['phytm'].append(phytm)
    out["aid_pp"].append(PPint - unppabs)
    out["attd_p"].append(Pint - unpabs)
    out["attd_ca"].append(Caint - uncaabs)
    out["attd_zn"].append(Znint - unznabs)

out = pd.DataFrame(out)



t = np.arange(sim_duration)
c = plt.cm.ocean(np.linspace(0, 0.85, no_sim))
xtick_pos = np.arange(0, 1440, 250)
ytick = [0, .1, .2, .3, .4, .5]

fig, ax = plt.subplots(2, 2, figsize=(7, 4.3), sharex=True)


for i in range(no_sim):
    df = scen[i]  # <-- le DataFrame du scénario i
    # for each min (0, 1440) I want to sum IDNPPs:blood{k} with k 0, nseg
    cols = [f'IDNPPs:blood{k}' for k in range(nseg)]
    IDNPPs_blood_sum = df[cols].sum(axis=1)*1000
    
    cols2 = [f'IDZns:blood{k}' for k in range(nseg)]
    IDZns_blood_sum = df[cols2].sum(axis=1)

    cols3 = [f'IDCas:blood{k}' for k in range(nseg)]
    IDCas_blood_sum = df[cols3].sum(axis=1)*1000

    cols4 = [f'IECas:blood{k}' for k in range(nseg)]
    IECas_blood_sum = df[cols4].sum(axis=1)*1000


    ax[0, 0].plot(t, IDNPPs_blood_sum, color=c[i], linewidth=1.5, linestyle  = ltype[i], label=labels[i])
    ax[0, 1].plot(t, IDZns_blood_sum, color=c[i], linewidth=1.5, linestyle  = ltype[i], label=labels[i])
    ax[1, 0].plot(t, IDCas_blood_sum, color=c[i], linewidth=1.5, linestyle  = ltype[i], label=labels[i])
    ax[1, 1].plot(t, IECas_blood_sum, color=c[i], linewidth=1.5, linestyle  = ltype[i], label=labels[i])

ax[0, 0].set_ylabel('Flux, mg/min')
ax[0, 0].set_ylim(0, 10)
ax[0, 0].set_yticks([0, 2, 4, 6, 8, 10])
ax[0, 0].set_yticklabels([str(i) for i in [0, 2, 4, 6, 8, 10]])
ax[0, 0].grid()

ax[0, 1].set_ylim(0, .20)
ax[0, 1].set_yticks([0, .05, .10, .15, .20, .25])
ax[0, 1].set_yticklabels([str(i) for i in [0, .05, .10, .15, .20, .25]])
ax[0, 1].grid()

#ax[1, 0].set_ylabel('Fluxes of absorption, mg/min')
ax[1, 0].set_xticks(xtick_pos)
ax[1, 0].set_xticklabels(xtick_pos)
ax[1, 0].set_xlim(0, 1440)

ax[1, 0].set_ylim(0, 8)
ax[1, 0].set_ylabel('Flux, mg/min')
ax[1, 0].set_yticks([0, 2.5, 5, 7.5, 10, 12.5])
ax[1, 0].set_yticklabels([str(i) for i in [0, 2.5, 5, 7.5, 10, 12.5]])
ax[1, 0].set_xlabel('Temps, min')
ax[1, 0].grid()

ax[1, 1].set_ylim(0, 2.0)
ax[1, 1].set_yticks([0, .5, 1.0, 1.50, 2.00, 2.5])
ax[1, 1].set_yticklabels([str(i) for i in [0, .5, 1.0, 1.50, 2.00, 2.5]])
ax[1, 1].set_xlabel('Temps, min')
ax[1, 1].grid()

handles, labels_leg = ax[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels_leg,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8,
    handlelength=2,
    columnspacing=1.2,
    alignment="center"
)

fig.subplots_adjust(bottom=0.22)
for ax, lab in zip(ax.ravel(), "ABCD"):
    ax.text(0.90, 0.92, lab, transform=ax.transAxes,
            ha="left", va="top", fontweight="light", fontsize=14)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure7.png', dpi= 300,  bbox_inches="tight")
plt.show()


