import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, HERE)
import ladtz_sim as L

def run(phi, nobj, ntr, seed=11):
    rng = np.random.default_rng(seed)
    names = ["baseline","ladtz_s38","ladtz_s15","ladtz_s8","pathttc_yaw","pathttc_lean"]
    inv = dict.fromkeys(names, 0)
    for _ in range(ntr):
        v, ph, k, objs, tc = L.scene(rng, phi, nobj)
        gt = int(np.argmin(tc))
        phi_m = ph + np.deg2rad(rng.normal(0,0.8)); k_yaw = k*(1+rng.normal(0,0.03))
        k_lean = L.G*np.tan(phi_m)/v**2
        sc = {n: [] for n in names}
        for o in objs:
            p=o[0]; vel=(o[1]-o[0])/L.DT
            r=np.linalg.norm(p)*(1+rng.normal(0,0.02)); th=np.arctan2(p[1],p[0])+np.deg2rad(rng.normal(0,1))
            pm=r*np.array([np.cos(th),np.sin(th)]); velm=vel+rng.normal(0,0.3,2)
            u=max(-(pm@(velm-np.array([v,0.])))/r,0)/r
            sc["baseline"].append(u)
            for s,name in ((None,"ladtz_s38"),(15,"ladtz_s15"),(8,"ladtz_s8")):
                pts=L.path_xy(k_yaw, v*np.arange(L.DT,L.T_LA+1e-9,L.DT)); mu=np.mean(np.arctan2(pts[:,1],pts[:,0]))
                sig=np.deg2rad(38+0.55*abs(np.rad2deg(phi_m)) if s is None else s)
                d=np.angle(np.exp(1j*(th-mu))); sc[name].append(u*(1+L.ALPHA*np.exp(-d**2/(2*sig**2))))
            for kk,name in ((k_yaw,"pathttc_yaw"),(k_lean,"pathttc_lean")):
                ego=L.path_xy(kk, v*L.TT); traj=pm[None]+velm[None]*L.TT[:,None]
                idx=np.nonzero(np.linalg.norm(traj-ego,axis=1)<L.DCONF)[0]
                sc[name].append(1/L.TT[idx[0]] if len(idx) and L.TT[idx[0]]>0 else (1e3 if len(idx) else 0))
        for n in names: inv[n]+=int(np.argmax(sc[n])!=gt)
    return {n: round(100*c/ntr,1) for n,c in inv.items()}
for phi in (0,20,40): print(phi, run(phi,5,400))
