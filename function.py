# **********************************************************************************************************************
# Julien Labarre, Christelle Loncke, Agnès Narcy, Maamer Jlali, Patrick Schlegel, Philippe Schmidely, 
# Marie-Pierre Létouneau-Montminy
# Modelling of calcium, phosphorus, and zinc fluxes in the gastrointestinal tract of growing pigs
# **********************************************************************************************************************

import pandas as pd 
import numpy as np
from scipy.interpolate import UnivariateSpline

def feeding(t, nmeals, dmi, meal_dur, feed_start, feed_end, dt):
    """Ingestion flux per minute at time t.""" 
    meal_input = np.zeros(len(t))

    # Feeding times
    if nmeals == 1:
        meal_starts = [feed_start]
    else:
        meal_interval = (feed_end - feed_start) / (nmeals - 1)
        meal_starts = [feed_start + i*meal_interval for i in range(nmeals)]
    
    # Ingestion flux during meals
    flux = (dmi / (nmeals * (meal_dur-1))) * dt
    
    for start in meal_starts:
        meal_input[(t >= start) & (t < start + meal_dur)] = flux
    return meal_input

def determine_phytase_eff(pH, act):
    """ Read the keff efficiency of microbial phytase"""
    # Use a look-up function to estimate the nonlinear fucntion
    u = UnivariateSpline(act.loc[:, 'pH'], act.loc[:, 'act']/100, k=3, s=0.0)
    return (u(pH))


def flux_equations(P, param, params_solp, meal_input, diet, dur, phytm_rel_act, phytv_rel_act, nseg) :
    """ Function to calculate the fluxes of the model """

    ''' Parameters '''
    # feeding parameters
    dmi = meal_input

    # parameters for npp relative bioavailaibility 
    ksp_msp = params_solp['ksp_msp']
    ksp_mcp = params_solp['ksp_mcp']
    ksp_mdcp = params_solp['ksp_mdcp']
    ksp_dcp = params_solp['ksp_dcp']
    ksp_nppv = params_solp['ksp_nppv']
    ksp_nppa = params_solp['ksp_nppa']


    # digesta passage parameters
    s_mrts = param['s_mrts'] 
    s_mrtl = param['s_mrtl']
    si_mrt = param['si_mrt']


    ''' Diet composition '''
    # non phytic phosphorus composition
    nppv = diet['nppv'] 
    mcp = diet['mcp'] * 0.23
    dcp = diet['dcp'] * 0.20
    msp = diet['msp'] * 0.20
    mdcp = diet['mdcp'] * 0.20 # à vérifier
    nppa = diet['nppa'] 
    nppo = diet['nppo']

    # phytic phosphorus 
    pp = diet['pp']

    # calcium
    cav = diet['cav']
    caa = diet['caa']
    cao = diet['cao']
    camcp = diet['mcp'] * 0.21
    cadcp = diet['dcp'] * 0.28
    limestone = diet['limestone'] * 0.385

    # zinc
    znv = diet['znv']
    znmin = diet['znmin']

    # phytase
    phytm = diet['phytm']
    phytv = diet['phytv']

    # Diet buffering capacity 
    kbbm = diet['BCfeed']

    ''' Transfer functions '''

    # Intake, EXT:Qs, f0
    f = [dmi]
    # Intake of NPPs, EXT:SDNPPs, f1
    f.append(dmi * (nppv * ksp_nppv + mcp * ksp_mcp + dcp * ksp_dcp + 
                    msp * ksp_msp + nppa * ksp_nppa + mdcp * ksp_mdcp + nppo))
    # Intake of NPPi, EXT:SDNPPi, f2
    f.append(dmi * (nppv * (1 - ksp_nppv) + mcp * (1 - ksp_mcp) + dcp * (1 - ksp_dcp) + 
                    msp * (1 - ksp_msp) + nppa * (1 - ksp_nppa) + mdcp * (1 - ksp_mdcp) + nppo))
    # Intake of NPPi, EXT:SDPPi, f3
    f.append(dmi * pp)
    # Intake of limestone, EXT:SDCaCO3, f4
    f.append(dmi * limestone)
    # Intake of other source of mineral Ca, EXT:SDCas, f5
    f.append(dmi * (camcp + cadcp + caa + cao))
    # Intake of Cav, EXT:SDCav, f6
    f.append(dmi * cav)
    # Intake of Znmin, EXT:SDZns, f7
    f.append(dmi * znmin)
    # Intake of Znv, EXT:SDZnv, f8
    f.append(dmi * znv)
    # Intake of microbial phytase, EXT:SDPHYTM, f9
    f.append(dmi * phytm)
    # Intake of vegetal phytase, EXT:SDPHYTV, f10
    f.append(dmi * phytv)
    # Intake of proton, EXT:SDH, f11
    f.append(dmi * 10**-diet['pHfood'] * 1000)

    # Gastric emptying, SDQs:EXT, f12
    f.append(P[0] * (1/s_mrts))
    # Gastric emptying, SDQl:EXT, f13
    f.append(P[1] * (1/s_mrts))
    # Gastric emptying, SDNPPs:IDNPPs, f14
    f.append(P[2] * (1/s_mrtl))
    # Gastric emptying, SDNPPi:IDNPPi, f15
    f.append(P[3] * (1/s_mrts))
    # Gastric emptying, SDPPs:IDPPs, f16
    f.append(P[4] * (1/s_mrtl))
    # Gastric emptying, SDPPi:IDPPi, f17
    f.append(P[5] * (1/s_mrts))
    # Gastric emptying, SDCaCO3:IDCas, f18
    f.append(P[6] * (1/s_mrts))
    # Gastric emptying, SDCas:IDCas, f19
    f.append(P[7] * (1/s_mrtl))
    # Gastric emptying, SDCav:IDCai, f20
    f.append(P[8] * (1/s_mrtl))
    # Gastric emptying, SDCai:IDCai, f21
    f.append(P[9] * (1/s_mrtl))
    # Gastric emptying, SDZns:IDZns, f22
    f.append(P[10] * (1/s_mrtl))
    # Gastric emptying, SDZnv:IDZni, f23
    f.append(P[11] * (1/s_mrts))
    # Gastric emptying, SDZni:IDZni, f24
    f.append(P[12] * (1/s_mrts))
    # Gastric emptying, SDPHYTM:EXT, f25
    f.append(P[13] * (1/s_mrtl))
    # Gastric emptying, SDPHYTV:EXT, f26
    f.append(P[14] * (1/s_mrtl))
    # Gastric emptying, SDH:EXT, f27
    f.append(P[15] * (1/s_mrtl))
    # Gastric emptying, SEF:EXT, f28
    f.append(P[16] * (1/s_mrtl))

    # Fluid secretion in the stomach, EXT:SDF, f29
    f.append(min(param['kgs'] * (((P[0] + P[1] + P[16])+ param["Vs0"])/param["Vsmax"]), param['kgs']))
    # Acid secretion by paritel cell, EXT:SQH, f30
    P[15] = max(P[15], 1e-8) 
    P[0] = max(P[0], 1e-8)
    P[1] = max(P[1], 1e-8)
    pHs = - np.log10(P[15]/((P[0] + P[1] + P[16])*1000)) if dur > 5 else diet['pHfood']
    #print(f"pH: {pHs}")
    Hgs = param['Hgs0'] + param['Hgsv'] * (1-10**-pHs/10**-2)
    f.append(min(Hgs * param['kgs'] * (((P[0] + P[1] + P[16])+ param["Vs0"])/param["Vsmax"]),  (param['Hgs0'] + param['Hgsv']) * param['kgs'] * (((P[0] + P[1] + P[16])+ param["Vs0"])/param["Vsmax"])))

    # Dissolution rate of feed, SDQs:SDQl, f31
    f.append(param["kd"] * P[0] * P[15])
    # Buffering effect of dissolution of feed, SDH:EXT, f32
    f.append(param["kd"] * P[0] * P[15] * kbbm)

    
    # Phytic phosphorus solublisation, SDPPi:SDPPs, f33
    f.append(param['kpp_sol'] * P[5])
    # Phytic phosphorus insolublisation, SDPPs:SDPPi, f34
    A_stp_sol_ppns = param['solpp_inf'] + ((param['solpp_0'] - param['solpp_inf'])/ 
                                      (1 + np.exp(- param['slope'] * (pHs - param['pH_inf']))))
    kpp_insol = min(((param['kpp_sol'] * A_stp_sol_ppns)/ (1 - A_stp_sol_ppns)), 1)
    f.append((kpp_insol * P[4]))

    if diet['phytm']==0 and diet['phytv'] == 0 : 
        f.append(0) # f36
        f.append(0) # f37
    else : 
        keffm =  max(min(determine_phytase_eff(pHs, phytm_rel_act),1),0)
        keffv =  max(min(determine_phytase_eff(pHs, phytv_rel_act),1),0)
        # Phytic phosphorus by mircobial phytase, SDPPs:SDNPPs_phytm, f35
        f.append(param['vmax_phytm'] * P[13] * P[4] * keffm/(param['km_phytm'] + P[4]))
        # Phytic phosphorus by mircobial phytase, SDPPs:SDNPPs_phytm, f36
        f.append(param['vmax_phytv'] * P[14] * P[4] * 0.6 * keffv / (param['km_phytv'] + P[4]))
    
    # Vegetal calcium solubilisation, SDCav:SDCas, f37
    f.append(max(min(((f[33]/param['w_p']/6) * param['ncapp'] * param['w_ca'] if pHs < 5 else 0), P[8]), 0))
    # Vegetal zinc solubilisation, SDZnv:SDZns, f38
    f.append(max(min(((f[33]/param['w_p']/6) * param['nznpp'] * 1000 * param['w_zn'] if pHs < 5 else 0), P[11]), 0))

    # For segment 1
    # Endogenous secretion 
    kpos = 0.01
    kneg = 0.01
    SEr = f[12] + f[13]
    POSp = kpos * SEr /1.8
    NEGp = kneg * P[17]
                  
    f.append(POSp - NEGp) # f39
    # Intestinal Endogenous phosphorus soluble in the intestine, EXT:IEPs, f40
    f.append(P[17] * param["rep"])
    # Intestinal Endogenous calcium soluble in the intestine, EXT:IECas0, f41
    f.append(P[17] * param["reca"])
    # Intestinal Endogenous zinc soluble in the intestine, EXT:IEZns0, f42
    f.append(P[17] * (param["reznb"] + (param["reznp"]* (diet["znmin"] + diet["znv"]))))

    # Intestinal transit, IDNPPs0:IDNPPs1, f43
    f.append(P[18] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDNPPi0:IDNPPsi1, f44
    f.append(P[19] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDPPs0:IDPPsi1, f45
    f.append(P[20] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDPPi0:IDPPi1, f46
    f.append(P[21] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDCas0:IDCas1, f47
    f.append(P[22] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDCai0:IDCai1, f48
    f.append(P[23] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDZns0:IDEns1, f49
    f.append(P[24] * (1/(param["si_mrt"]/nseg)))
    # Intestinal transit, IDZni0:IDZni1, f50
    f.append(P[25] * (1/(param["si_mrt"]/nseg)))
    # Instestinal transit, IEPs0:IEPs1, f51
    f.append(P[26] * (1/(param["si_mrt"]/nseg)))
    # Instestinal transit, IEPi0:IEPi1, f52
    f.append(P[27] * (1/(param["si_mrt"]/nseg)))
    # Instestinal transit, IECas0:IECas1, f53
    f.append(P[28] * (1/(param["si_mrt"]/nseg)))
    # Instestinal transit, IECai0:IECai1, f54
    f.append(P[29] * (1/(param["si_mrt"]/nseg)))
    # Instestinal transit, IEZns0:IEZns1, f55
    f.append(P[30] * (1/(param["si_mrt"]/nseg)))
    # Instestinal transit, IEZni0:IEZni1, f56
    f.append(P[31] * (1/(param["si_mrt"]/nseg)))

    # Absorption of IDNPPs, IDNPPsj:blood0, f57
    f.append(max(min((P[18] * param["vmaxp"] / (param["kmp"] + P[18] + P[26]) + param["kap"] * P[18]), P[18]), 0))
    # Absorption of IEPs, IEPsj:blood0, f58
    f.append(max(min((P[26] * param["vmaxp"] / (param["kmp"] + P[18] + P[26]) + param["kap"] * P[26]), P[26]), 0))
    # Absorption of IDCas, IDCasj:blood0, f59
    f.append(max(min((P[22] * param["vmaxca"] / (param["kmca"] + P[22] + P[28]) + param["kaca"] * P[22]), P[22]), 0))
    # Absorption of IECas, IECasj:blood0, f60
    f.append(max(min((P[28] * param["vmaxca"] / (param["kmca"] + P[22] + P[28]) + param["kaca"] * P[28]), P[28]), 0))
    # Absorption of IDZns, IDZnsj:blood0, f61
    f.append(max(min((P[24] * param["vmaxzn"] / (param["kmzn"] + P[24] + P[30]) + param["kazn"] * P[24]), P[24]), 0))
    # Absorption of IEZns, IEZnsj:blood0, f62
    f.append(max(min((P[30] * param["vmaxzn"] / (param["kmzn"] + P[24] + P[30]) + param["kazn"] * P[30]), P[30]), 0))

    # Desquamation of endogenous Zn, EXT:IEZNi0, f63
    f.append(param["reznd"])
    # Insolubilisation of dietary phytic phosphorus, IDPPsj:IDPPij, f64
    f.append(param['kpp_insolsi'] * P[20])
    # Hydrolysis of dietary phytic phosphorus, IDPPsj:IDNPPsj, f65
    f.append(param["kpp_hsi"] * P[20])
    # Insolubilisation of dietary calcium by phytic phosphorus, IDCasj:IDCaij, f66
    f.append(max(min((((f[17] + param["kpp_hsi"] * P[20])/6)/param["w_p"]) * param['ncapp'] * param["w_ca"] , P[22]), 0) if P[22]>0 else 0)
    # Insolubilisation of endogenous calcium by phytic phosphorus, IECasj:IECaij, f67
    f.append(max(min((((param["kpp_hsi"] * P[20])/6)/param["w_p"]) * param['ncapp'] * param["w_ca"], P[28]), 0) if P[28]>0 else 0)
    # Insolubilisation of dietary zinc by phytic phosphorus, IDZnsj:IDZnij, f68
    f.append(max(min((((f[17] + param["kpp_hsi"] * P[20])/6)/param["w_p"]) * param["nznpp"] * 1000 * param["w_zn"] , P[24]), 0) if P[24]>0 else 0)
    # Insolubilisation of endogenous zinc by phytic phosphorus, IEZnsj:IEZnij, f69
    f.append(max(min((((param["kpp_hsi"] * P[20])/6)/param["w_p"]) * param["nznpp"] * 1000 * param["w_zn"], P[30]), 0) if P[30]>0 else 0)

    # Formation of insoluble complexes Ca-P complexes, IDCas:ICais, f70
    CaP_ca0 = param["rcanpp"] * (P[22]+P[28]) if P[22]>0 else 0
    f.append(CaP_ca0 * (P[22]/(P[22]+P[28])) if P[22]>0 else 0)
    # Formation of insoluble complexes Ca-P complexes, IECas:IEais, f71
    f.append(CaP_ca0 * (P[28]/(P[22]+P[28])) if P[22]>0 else 0)
    
    # Formation of insoluble complexes Ca-P complexes, IDNNPs:IDNNPi, f72
    f.append(min(max(CaP_ca0 /param['w_ca'] * param['ncap'] * param["w_p"] * (P[18]/(P[18]+P[26])), 0), P[18]) if P[18]>0 else 0)
    # Formation of insoluble complexes Ca-P complexes, IEPs:IEPi, f73
    f.append(min(max(CaP_ca0 /param['w_ca'] * param['ncap'] * param["w_p"] * (P[26]/(P[18]+P[26])), 0), P[26])if P[18]>0 else 0)

    # Segment n 
    add_seg = range(1, nseg)

    for j in add_seg : 

        # Define the pools 
        IDNPPsj = P[18 + 14 * j] # Intestinal Dietary Non Phytic Phosphorus soluble in the intestine, IDNPPsj
        IDNPPij = P[19 + 14 * j] # Intestinal Dietary Non Phytic Phosphorus insoluble in the intestine, IDNPPij
        IDPPsj = P[20 + 14 * j] # Intestinal Dietary Phytic Phosphorus soluble in the intestine, IDPPsj
        IDPPij = P[21 + 14 * j] # Intestinal Dietary Phytic Phosphorus insoluble in the intestine, IDPPij
        IDCasj = P[22 + 14 * j] # Intestinal Dietary Calcium soluble in the intestine, IDCasj
        IDCaij = P[23 + 14 * j] # Intestinal Dietary Calcium insoluble in the intestine, IDCaij
        IDZnsj = P[24 + 14 * j] # Intestinal Dietary Zinc soluble in the intestine, IDZnsj
        IDZnij = P [25 + 14 * j] # Intestinal Dietary Zinc insoluble in the intestine, IDZni0
        IEPsj = P[26 + 14 * j] # Intestinal Endogenous phosphorus soluble in the intestine, IEPsj
        IEPij = P[27 + 14 * j] # Intestinal Endogenous phosphorus insoluble in the intestine, IEPij
        IECasj = P[28 + 14 * j] # Intestinal Endogenous calcium soluble in the intestine, IECasj
        IECaij = P[29 + 14 * j] # Intestinal Endogenous calcium insoluble in the intestine, IECaij
        IEZnsj = P[30 + 14 * j] # Intestinal Endogenous zinc soluble in the intestine, IEZnsj
        IEZnij = P[31 + 14 * j] # Intestinal Endogenous zinc insoluble in the intestine, IEZnij

        # Intestinal transit, IDNPPs0:IDNPPs1, f74
        f.append(IDNPPsj * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDNPPi0:IDNPPsi1, f75
        f.append(IDNPPij * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDPPs0:IDPPsi1, f76
        f.append(IDPPsj * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDPPi0:IDPPi1, f77
        f.append(IDPPij * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDCas0:IDCas1, f78
        f.append(IDCasj * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDCai0:IDCai1, f79
        f.append(IDCaij * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDZns0:IDEns1, f80
        f.append(IDZnsj * (1/(param["si_mrt"]/nseg)))
        # Intestinal transit, IDZni0:IDZni1, f81
        f.append(IDZnij * (1/(param["si_mrt"]/nseg)))
        # Instestinal transit, IEPs0:IEPs1, f82
        f.append(IEPsj * (1/(param["si_mrt"]/nseg)))
        # Instestinal transit, IEPi0:IEPi1, f83
        f.append(IEPij * (1/(param["si_mrt"]/nseg)))
        # Instestinal transit, IECas0:IECas1, f84
        f.append(IECasj * (1/(param["si_mrt"]/nseg)))
        # Instestinal transit, IECai0:IECai1, f85
        f.append(IECaij * (1/(param["si_mrt"]/nseg)))
        # Instestinal transit, IEZns0:IEZns1, f86
        f.append(IEZnsj * (1/(param["si_mrt"]/nseg)))
        # Instestinal transit, IEZni0:IEZni1, f87
        f.append(IEZnij * (1/(param["si_mrt"]/nseg)))

        # Absorption of IDNPPs, IDNPPsj:blood0, f88
        f.append(max(min((IDNPPsj * param["vmaxp"] / (param["kmp"] + IDNPPsj + IEPsj) + param["kap"] * IDNPPsj), IDNPPsj), 0))
        # Absorption of IEPs, IEPsj:blood0, f89
        f.append(max(min((IEPsj * param["vmaxp"] / (param["kmp"] + IDNPPsj + IEPsj) + param["kap"] * IEPsj), IEPsj), 0))
        # Absorption of IDCas, IDCasj:blood0, f90
        f.append(max(min((IDCasj * param["vmaxca"] / (param["kmca"] + IDCasj + IECasj) + param["kaca"] * IDCasj), IDCasj), 0))
        # Absorption of IECas, IECasj:blood0, f91
        f.append(max(min((IECasj * param["vmaxca"] / (param["kmca"] + IDCasj + IECasj) + param["kaca"] * IECasj), IECasj), 0))
        # Absorption of IDZns, IDZnsj:blood0, f92
        f.append(max(min((IDZnsj * param["vmaxzn"] / (param["kmzn"] + IDZnsj + IEZnsj) + param["kazn"] * IDZnsj), IDZnsj), 0))
        # Absorption of IEZns, IEZnsj:blood0, f93
        f.append(max(min((IEZnsj * param["vmaxzn"] / (param["kmzn"] + IDZnsj + IEZnsj) + param["kazn"] * IEZnsj), IEZnsj), 0))

        # Desquamation of endogenous Zn, EXT:IEZNi0, f94
        f.append(param["reznd"])
        # Insolubilisation of dietary phytic phosphorus, IDPPsj:IDPPij, f95
        f.append(param['kpp_insolsi'] * IDPPsj)
        # Hydrolysis of dietary phytic phosphorus, IDPPsj:IDNPPsj, f96
        f.append(param["kpp_hsi"] * IDPPsj)
        # Insolubilisation of dietary calcium by phytic phosphorus, IDCasj:IDCaij, f97
        f.append(max(min(((param["kpp_hsi"] * IDPPsj/6)/param["w_p"]) * param['ncapp'] * param["w_ca"] * IDCasj/(IDCasj+IECasj), IDCasj), 0) if IDCasj>0 else 0)
        # Insolubilisation of endogenous calcium by phytic phosphorus, IECasj:IECaij, f98
        f.append(max(min(((param["kpp_hsi"] * IDPPsj/6)/param["w_p"]) * param['ncapp'] * param["w_ca"] * IECasj/(IDCasj+IECasj), IECasj), 0) if IECasj>0 else 0)
        # Insolubilisation of dietary zinc by phytic phosphorus, IDZnsj:IDZnij, f99
        f.append(max(min(((param["kpp_hsi"] * IDPPsj/6)/param["w_p"]) * param["nznpp"] * param["w_zn"] * 1000 * IDZnsj/(IDZnsj+IEZnsj), IDZnsj), 0) if IDZnsj>0 else 0)
        # Insolubilisation of endogenous zinc by phytic phosphorus, IEZnsj:IEZnij, f100
        f.append(max(min(((param["kpp_hsi"] * IDPPsj/6)/param["w_p"]) * param["nznpp"] * param["w_zn"] * 1000 * IEZnsj/(IDZnsj+IEZnsj), IEZnsj), 0) if IEZnsj>0 else 0)
        # Formation of insoluble complexes Ca-P complexes, IDCas:ICais, f101
        CaP_ca0 = param["rcanpp"] * (IDCasj + IECasj) if IDCasj>0 else 0
        f.append(CaP_ca0 * (IDCasj /(IDCasj + IECasj)) if IDCasj>0 else 0)
        # Formation of insoluble complexes Ca-P complexes, IECas:IEais, f102
        f.append(CaP_ca0 * (IECasj/(IDCasj + IECasj)) if IECasj > 0 else 0)
        
        # Formation of insoluble complexes Ca-P complexes, IDNNPs:IDNNPi, f103
        f.append(min(max(CaP_ca0 /param['w_ca'] * param['ncap'] * param["w_p"] * (IDNPPsj/(IDNPPsj + IEPsj)), 0), IDNPPsj) if IDNPPsj>0 else 0)
        # Formation of insoluble complexes Ca-P complexes, IEPs:IEPi, f104
        f.append(min(max(CaP_ca0 /param['w_ca'] * param['ncap'] * param["w_p"] * (IEPsj/(IDNPPsj + IEPsj)), 0), IEPsj) if IEPsj>0 else 0)

    return f

def differential_eq(t, P, param, params_solp, meal_input, diet, dur, phytm_rel_act, phytv_rel_act, nseg):
    """Function containing the differential equations"""
    # Size of initial vector depend on number of segments
    
    equ = np.array(P)
    #print(equ)
    # Call the function that contains the fluxes
    f = flux_equations(P, param, params_solp, meal_input, diet, dur, phytm_rel_act, phytv_rel_act, nseg) 

    # Stomach Feed, SDQs
    equ[0] = f[0] * 0.8 - f[12] - f[31]
    # Stomach Feed solubilisation, SDQl
    equ[1] = f[0] * 0.2  - f[13] + f[31]
    # Stomach Dietary Non Phytic Phosphorus soluble, SDNPPs
    equ[2] = f[1] - f[14] + f[35] + f[36]
    # Stomach Dietary Non Phytic Phosphorus insoluble, SDNPPi
    equ[3] = f[2] - f[15]
    # Stomach Dietary Phytic Phosphorus soluble, SDPPs
    equ[4] = 0 - f[16] + f[33] - f[34] - f[35] - f[36]
    # Stomach Dietary Phytic Phosphorus insoluble, SDPPi
    equ[5] = f[3] - f[17] - f[33] + f[34]
    # Stomach Dietary calcium carbonatre, SDCaCO3
    equ[6] = f[4] - f[18] 
    # Stomach Dietary calcium soluble, SDCas
    equ[7] = f[5] - f[19] + f[37]
    # Stomach Dietary calcium vegetal, SDCav
    equ[8] = f[6] - f[20] - f[37]
    # Stomach Dietary calcium insoluble, SDCai
    equ[9] = - f[21]
    # Stomach Dietary zinc soluble, SDZns
    equ[10] = f[7] + f[38] - f[22] 
    # Stomach Dietary zinc vegetal, SDZnv
    equ[11] = f[8] - f[23] - f[38]
    # Stomach Dietary zinc insoluble, SDZni
    equ[12] = - f[24]
    # Stomach Dietary microbial phytase, SDPHYTM
    equ[13] = f[9] - f[25]
    # Stomach Dietary vegetal phytase, SDPHYTV
    equ[14] = f[10] - f[26]
    # Stomach Dietary proton, SDH
    equ[15] = f[11] - f[27] + f[30] - f[32]
    # Stomach Fluid, SEF
    equ[16] = 0 - f[28] + f[29]
    # Pancreatic and Biliary Secretion Rate, PBRS
    equ[17] = f[39]
    # For segment 1
    # Intestinal Dietary Non Phytic Phosphorus soluble in the intestine, IDNPPs0
    equ[18] = f[14] - f[43] - f[57] + f[65] - f[72]
    # Intestinal Dietary Non Phytic Phosphorus insoluble in the intestine, IDNPPi0
    equ[19] = f[15] - f[44] + f[72]
    # Intestinal Dietary Phytic Phosphorus soluble in the intestine, IDPPs0
    equ[20] = f[16] + f[17] - f[45] - f[64] - f[65]
    # Intestinal Dietary Phytic Phosphorus insoluble in the intestine, IDPPi0
    equ[21] = - f[46] + f[64] 
    # Intestinal Dietary Calcium soluble in the intestine, IDCas0
    equ[22] = f[18] + f[19] - f[47] - f[59] - f[66] - f[70]
    # Intestinal Dietary Calcium insoluble in the intestine, IDCai0
    equ[23] = f[20] + f[21] - f[48] + f[66] + f[70]
    # Intestinal Dietary Zinc soluble in the intestine, IDZns0
    equ[24] = f[22] - f[49] - f[61] - f[68]
    # Intestinal Dietary Zinc insoluble in the intestine, IDZni0
    equ[25] = f[23] - f[50] + f[68]
    # Intestinal Endogenous phosphorus soluble in the intestine, IEPs0
    equ[26] = f[40] - f[51] - f[58] - f[73]
    # Intestinal Endogenous phosphorus insoluble in the intestine, IEPi0
    equ[27] = 0 - f[52] + f[73]
    # Intestinal Endogenous calcium soluble in the intestine, IECas0
    equ[28] = f[41] - f[53] - f[60] - f[67] - f[71]
    # Intestinal Endogenous calcium insoluble in the intestine, IECai0
    equ[29] = 0 - f[54] + f[67] + f[71]
    # Intestinal Endogenous zinc soluble in the intestine, IEZns0
    equ[30] = f[42] - f[55] - f[62] + f[63] - f[69]
    # Intestinal Endogenous zinc insoluble in the intestine, IEZni0
    equ[31] = 0 - f[56] + f[69]

    # For all the other segments
    add_seg = range(0, nseg - 1)

    for i in add_seg : 
        # Intestinal Dietary Non Phytic Phosphorus soluble in the intestine, IDNPPs0
        equ[32 + 14 * i] = f[43 + 31 * i] - f[74 + 31 * i] - f[88 + 31 * i] + f[96 + 31 * i] - f[103 + 31 * i]
        # Intestinal Dietary Non Phytic Phosphorus insoluble in the intestine, IDNPPi0
        equ[33 + 14 * i] = f[44 + 31 * i] - f[75 + 31 * i] + f[103 + 31 * i]
        # Intestinal Dietary Phytic Phosphorus soluble in the intestine, IDPPs0
        equ[34 + 14 * i] = f[45 + 31 * i] - f[76 + 31 * i] - f[95 + 31 * i] - f[96 + 31 * i]
        # Intestinal Dietary Phytic Phosphorus insoluble in the intestine, IDPPi0
        equ[35 + 14 * i] = f[46 + 31 * i] - f[77 + 31 * i] + f[95 + 31 * i]
        # Intestinal Dietary Calcium soluble in the intestine, IDCas0
        equ[36 + 14 * i] = f[47 + 31 * i] - f[78 + 31 * i] - f[90 + 31 * i] - f[97 + 31 * i] - f[101 + 31 * i]
        # Intestinal Dietary Calcium insoluble in the intestine, IDCai0
        equ[37 + 14 * i] = f[48 + 31 * i] - f[79 + 31 * i] + f[97 + 31 * i] + f[101 + 31 * i]
        # Intestinal Dietary Zinc soluble in the intestine, IDZns0
        equ[38 + 14 * i] = f[49 + 31 * i] - f[80 + 31 * i] - f[92 + 31 * i] - f[99 + 31 * i]
        # Intestinal Dietary Zinc insoluble in the intestine, IDZni0
        equ[39 + 14 * i] = f[50 + 31 * i] - f[81 + 31 * i] + f[99 + 31 * i]
        # Intestinal Endogenous phosphorus soluble in the intestine, IEPs0
        equ[40 + 14 * i] = f[51 + 31 * i] - f[82 + 31 * i] - f[89 + 31 * i] - f[104 + 31 * i]
        # Intestinal Endogenous phosphorus insoluble in the intestine, IEPi0
        equ[41 + 14 * i] = f[52 + 31 * i] - f[83 + 31 * i] + f[104 + 31 * i]
        # Intestinal Endogenous calcium soluble in the intestine, IECas0
        equ[42 + 14 * i] = f[53 + 31 * i] - f[84 + 31 * i] - f[91 + 31 * i] - f[98 + 31 * i] - f[102 + 31 * i]
        # Intestinal Endogenous calcium insoluble in the intestine, IECai0
        equ[43 + 14 * i] = f[54 + 31 * i] - f[85 + 31 * i] + f[98 + 31 * i] + f[102 + 31 * i]
        # Intestinal Endogenous zinc soluble in the intestine, IEZns0
        equ[44 + 14 * i] = f[55 + 31 * i] - f[86 + 31 * i] - f[93 + 31 * i] + f[94 + 31 * i] - f[100 + 31 * i]
        # Intestinal Endogenous zinc insoluble in the intestine, IEZni0
        equ[45 + 14 * i] = f[56 + 31 * i] - f[87 + 31 * i] + f[100 + 31 * i]

    return equ

def calculate_pH(df):
    df.loc[:, 'pH'] = -np.log10(df['SDH'] / ((df['SDQs'] + df['SDQl'] + df['SEF']) * 1000))    
    return df

def name_pools(df, nseg):
    "Fucntion to rename pools of the model"
    cn = ["SDQs", "SDQl", "SDNPPs", "SDNPPi", "SDPPs", "SDPPi", "SDCaCO3", "SDCas", "SDCav", "SDCai",
        "SDZns", "SDZnv" ,"SDZni", "SDPHYTM", "SDPHYTV", "SDH", "SEF", 'PBSr']
    
    for k in range(0, nseg):
        cn.append(f"IDNPPs{k}")
        cn.append(f"IDNPPi{k}")
        cn.append(f"IDPPs{k}")
        cn.append(f"IDPPi{k}")
        cn.append(f"IDCas{k}")
        cn.append(f"IDCai{k}")
        cn.append(f"IDZns{k}")
        cn.append(f"IDZni{k}")
        cn.append(f"IEPs{k}")
        cn.append(f"IEPi{k}")
        cn.append(f"IECas{k}")
        cn.append(f"IECai{k}")
        cn.append(f"IEZns{k}")
        cn.append(f"IEZni{k}")
    
    df.columns = cn
    return df

def names_fluxes(dfluxes, nseg):
    """Pools + new flux layout: fixed stomach, iterative segments 1..N (last -> out)."""

    cols = [
        "EXT:Qs", "EXT:SDNPPs", "EXT:SDNPPi", "EXT:SDPPi", "EXT:SDCaCO3", "EXT:SDCas", 
        "EXT:SDCav", "EXT:SDZns", "EXT:SDZnv", "EXT:SDPHYTM", "EXT:SDPHYTV", "EXT:SDH", 
        "Qs:SI1", "Ql:SI1", "SDNPPs:IDNPPs0", "SDNPPi:IDNPPi0", "SDPPs:IDPPs0", "SDPPi:IDPPi0", 
        "SDCaCO3:IDCas0", "SDCas:IDCas0", "SDCav:IDCai0", "SDCai:IDCai0", "SDZns:IDZns0", 
        "SDZnv:IDZni0", "SDZni:IDZni0", "SDPHYTM:EXT", "SDPHYTV:EXT", "SDH:EXT", "SEF:EXT",
        "EXT:SEF", "EXT:SQH", "SDQs:SDQl", "SDH_EXT_buffer", "SDPPi:SDDPs", "SDPPs:SDPPi",
        "SDPPs:SDNPPs_phytm", "SDPPs:SDNPPs_phytv", "SDCav:SDCas", "SDZnv:SDZns",
        "PBSr_rate", "EXT:IEPs0", "EXT:IECas0", "EXT:IEZns0"
    ]

    for j in range(0, nseg):

        cols.append(f"IDNPPs:IDNPPs{j}")
        cols.append(f"IDNPPi:IDNPPi{j}")
        cols.append(f"IDPPs:IDPPs{j}")
        cols.append(f"IDPPi:IDPPi{j}")
        cols.append(f"IDCas:IDCas{j}")
        cols.append(f"IDCai:IDCai{j}")
        cols.append(f"IDZns:IDZns{j}")
        cols.append(f"IDZni:IDZni{j}")
        cols.append(f"IEPs:IEPs{j}")
        cols.append(f"IEPi:IEPi{j}")
        cols.append(f"IECas:IECas{j}")
        cols.append(f"IECai:IECai{j}")
        cols.append(f"IEZns:IEZns{j}")
        cols.append(f"IEZni:IEZni{j}")
        cols.append(f"IDNPPs:blood{j}") 
        cols.append(f"IEPs:blood{j}") 
        cols.append(f"IDCas:blood{j}") 
        cols.append(f"IECas:blood{j}") 
        cols.append(f"IDZns:blood{j}") 
        cols.append(f"IEZns:blood{j}") 
        cols.append(f"Desq:IEZns{j}")
        cols.append(f"IDPPs:IDPPi{j}")
        cols.append(f"IDPPs:IDNPPs{j}")
        cols.append(f"IDCas:IDCai_CaPP{j}")
        cols.append(f"IECas:IECai_CaPP{j}")
        cols.append(f"IDZns:IDZni{j}")
        cols.append(f"IEZns:IEZni{j}")
        cols.append(f"IDCas:IDCai_CaP{j}")
        cols.append(f"IECas:IECai_CaP{j}")
        cols.append(f"IDNPPs:IDNPPi_CaP{j}")
        cols.append(f"IEPs:IEPi_CaP{j}")

    dfluxes.columns = cols
    return dfluxes

def calculate_fluxes(df, param, params_solp, meal_input, diet, dur, phytm_rel_act, phytv_rel_act, nseg, time_col="Time"):
    """Compute fluxes from model pools."""
    db = []
    for i in range(0,len(df.index)):
        row = df.iloc[i]
        t = row[time_col] if time_col in df.columns else df.index[i]
        row = row.iloc[1:]
        f = flux_equations(np.array(row), param, params_solp, meal_input[i], diet, dur, phytm_rel_act, phytv_rel_act, nseg)
        db.append(f)
    return pd.DataFrame(db)

def sum_col_un(df, col, nseg):
    """Sum values per fluxes, for the whole simulation"""
    suma = 0
    for i in col:
        column_name = f'{i}:{i}{nseg-1}'
        if column_name in df.columns:
            column_sum = df[column_name].sum()
            #print(f"Sum of {column_name}: {column_sum}")
            suma += column_sum
        else:
            print(f"Column {column_name} not found in DataFrame.")
    #print(f"Total sum: {suma}")
    return suma

def true_absorption(df, n_seg, flux):
    """Calculate percentage of absorbed AA"""
    suma = []
    abso = 0
    for i in range(1, n_seg):
        suma.append(df[f'{flux}{i}'].sum())
        abso += df[f'{flux}{i}'].sum()
    return abso


def absorption(df, sim_duration, nseg, flux):
    """Plot the fluxes of absorption """
    suma = []
    for i in range(1, sim_duration):
        abso = 0
        for k in range(0, nseg-1):
            abso += df[i, f'{flux}{k}'].sum()
        suma.append(abso)
    return suma


def create_df_seg(df_seg, df_final, nseg, flux):
    """
    Create a dataframe with the sum of a given flux across time, per segment.
    
    Parameters
    ----------
    df_seg : pd.DataFrame
        Base dataframe (usually time column).
    df_final : pd.DataFrame
        Final dataframe with fluxes and pools.
    nseg : int
        Number of segments.
    flux : str
        Flux prefix to extract (e.g. "f_IDNPPs").
    """
    # Collect flux columns per segment
    flux_cols = [f"{flux}{k}" for k in range(1, nseg)]
    arr = df_final[flux_cols].to_numpy()

    # Sum over time for each segment
    sums = arr.sum(axis=0)

    # Build result dataframe
    df_f = pd.DataFrame({
        flux: sums,
        "Segment": np.arange(1, nseg)
    })
    return df_f


