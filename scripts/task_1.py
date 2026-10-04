import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

np.random.seed(42)

# CSTR Physical Parameters
V = 100.0          # Reactor Volume [L]
k0 = 7.2e10        # Pre-exponential factor [1/min]
E_over_R = 8750.0  # Activation energy parameter [K]
delta_H = -5.0e4   # Heat of reaction [J/mol] (exothermic)
rho_Cp = 1000.0    # Density * Heat Capacity [J / (L*K)]
UA_nominal = 5.0e4 # Nominal heat transfer coefficient [J / (min*K)]

T_cutoff = 400.0   # Explosion / runaway threshold [K]
R_max = 20.0       # Warning horizon limit [min]

def cstr_rhs(t, y, q, cAi, Ti, Tc, UA):
    cA, T = y
    cA = max(cA, 1e-7)
    T = max(T, 150.0)
    
    k = k0 * np.exp(-E_over_R / T)
    rA = k * cA
    
    dcA_dt = (q / V) * (cAi - cA) - rA
    dT_dt = (q / V) * (Ti - T) + ((-delta_H) / rho_Cp) * rA - (UA / (V * rho_Cp)) * (T - Tc)
    return [dcA_dt, dT_dt]

def simulate_single_run(run_id, t_max=40.0, dt=0.2):
    q_base = np.random.uniform(95.0, 105.0)
    cAi_base = np.random.uniform(0.95, 1.05)
    Ti_base = np.random.uniform(345.0, 355.0)
    Tc_base = np.random.uniform(295.0, 305.0)
    UA_base = UA_nominal
    
    cA_curr = np.random.uniform(0.45, 0.55)
    T_curr = np.random.uniform(348.0, 355.0)
    
    is_upset = np.random.rand() > 0.25
    upset_time = np.random.uniform(5.0, 12.0)
    scenario = np.random.choice(["coolant_failure", "jacket_warm", "feed_surge", "flow_drop"])
    
    q, cAi, Ti, Tc, UA = q_base, cAi_base, Ti_base, Tc_base, UA_base
    
    times = []
    q_list, cAi_list, Ti_list, Tc_list = [], [], [], []
    cA_list, T_list, dT_dt_list = [], [], []
    
    t_cutoff_hit = None
    t_eval = np.arange(0.0, t_max, dt)
    
    for t in t_eval:
        if is_upset and (t >= upset_time):
            if scenario == "coolant_failure":
                UA = UA_base * np.random.uniform(0.15, 0.35)
            elif scenario == "jacket_warm":
                Tc = Tc_base + np.random.uniform(25.0, 45.0)
            elif scenario == "feed_surge":
                cAi = cAi_base * np.random.uniform(1.3, 1.7)
            elif scenario == "flow_drop":
                q = q_base * np.random.uniform(0.4, 0.6)
                
        derivs = cstr_rhs(t, [cA_curr, T_curr], q, cAi, Ti, Tc, UA)
        dT_dt_curr = derivs[1]
        
        times.append(round(t, 2))
        q_list.append(round(q, 2))
        cAi_list.append(round(cAi, 4))
        Ti_list.append(round(Ti, 2))
        Tc_list.append(round(Tc, 2))
        cA_list.append(round(cA_curr, 5))
        T_list.append(round(T_curr, 2))
        dT_dt_list.append(round(dT_dt_curr, 4))
        
        if T_curr >= T_cutoff:
            t_cutoff_hit = t
            break
            
        sol = solve_ivp(
            cstr_rhs,
            [t, t + dt],
            [cA_curr, T_curr],
            args=(q, cAi, Ti, Tc, UA),
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )
        cA_curr = sol.y[0][-1]
        T_curr = sol.y[1][-1]
        
    df = pd.DataFrame({
        "run_id": run_id,
        "time": times,
        "q": q_list,
        "cAi": cAi_list,
        "Ti": Ti_list,
        "Tc": Tc_list,
        "cA": cA_list,
        "T": T_list,
        "dT_dt": dT_dt_list
    })
    
    if t_cutoff_hit is not None:
        remaining = t_cutoff_hit - df["time"]
        df["time_to_cutoff"] = np.minimum(R_max, remaining).clip(lower=0.0).round(2)
        df["is_runaway"] = 1
    else:
        df["time_to_cutoff"] = R_max
        df["is_runaway"] = 0
        
    return df

num_trajectories = 100
all_runs = [simulate_single_run(i) for i in range(1, num_trajectories + 1)]
full_dataset = pd.concat(all_runs, ignore_index=True)

csv_filename = "cstr_runaway_trajectories.csv"
full_dataset.to_csv(csv_filename, index=False)

print(f"Generated {len(full_dataset)} rows across {num_trajectories} trajectories.")
print(f"Saved to: {csv_filename}")