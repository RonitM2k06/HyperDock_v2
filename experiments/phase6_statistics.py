import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def cliffs_delta(x, y):
    """Calculates Cliff's Delta effect size."""
    x = np.asarray(x)
    y = np.asarray(y)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0.0
    mat = np.sign(x[:, None] - y)
    return mat.sum() / (m * n)

def calc_spacing(front):
    """Spacing metric (Schott 1995) for 2D front."""
    if len(front) < 2: return 0.0
    sorted_front = front[np.argsort(front[:, 0])]
    dists = np.linalg.norm(sorted_front[:-1] - sorted_front[1:], axis=1)
    mean_dist = np.mean(dists)
    return np.sqrt(np.sum((dists - mean_dist)**2) / (len(front)-1))

def calc_spread(front, ideal, nadir):
    """Spread (Delta) metric for 2D front."""
    if len(front) < 2: return 0.0
    sorted_front = front[np.argsort(front[:, 0])]
    df = np.linalg.norm(sorted_front[0] - ideal)
    dl = np.linalg.norm(sorted_front[-1] - nadir)
    dists = np.linalg.norm(sorted_front[:-1] - sorted_front[1:], axis=1)
    dm = np.mean(dists)
    return (df + dl + np.sum(np.abs(dists - dm))) / (df + dl + len(dists) * dm)

def main():
    exp_dir = os.path.dirname(__file__)
    res_path = os.path.join(exp_dir, 'results.csv')
    conv_path = os.path.join(exp_dir, 'convergence.csv')
    pareto_dir = os.path.join(exp_dir, 'pareto_fronts')
    tables_dir = os.path.join(exp_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    if not os.path.exists(res_path):
        print("Missing results.csv")
        return
        
    df = pd.read_csv(res_path)
    
    # Pre-process baseline performance into pseudo-pareto fronts
    # Baselines just return one point.
    baseline_fronts = {}
    for idx, row in df[df['algorithm'] != 'NSGA-II'].iterrows():
        ds = row['dataset']
        algo = row['algorithm']
        pt = [row['com_deviation'], -row['casc_score']] # casc_score in df is already positive, so neg_casc = -casc_score
        if ds not in baseline_fronts:
            baseline_fronts[ds] = {}
        baseline_fronts[ds][algo] = np.array([pt])

    datasets = df['dataset'].unique()
    
    all_metrics = []
    
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        nsga2_runs = ds_df[ds_df['algorithm'] == 'NSGA-II']
        
        # 1. Collect all points for normalization
        all_points = []
        for algo in ['FF', 'BFD-V', 'BFD-P']:
            if ds in baseline_fronts and algo in baseline_fronts[ds]:
                all_points.append(baseline_fronts[ds][algo][0])
                
        nsga2_fronts = {}
        for seed in nsga2_runs['seed']:
            pf_file = os.path.join(pareto_dir, f"{ds}_seed_{seed}.csv")
            if os.path.exists(pf_file):
                pf = pd.read_csv(pf_file).values
                nsga2_fronts[seed] = pf
                for pt in pf:
                    all_points.append(pt)
                    
        all_points = np.array(all_points)
        if len(all_points) == 0:
            continue
            
        f_min = np.min(all_points, axis=0)
        f_max = np.max(all_points, axis=0)
        denom = np.where(f_max - f_min == 0, 1e-9, f_max - f_min)
        
        # True Pareto Front for IGD
        nds = NonDominatedSorting().do(all_points, only_non_dominated_front=True)
        true_pf = all_points[nds]
        true_pf_norm = (true_pf - f_min) / denom
        
        hv_ind = HV(ref_point=np.array([1.1, 1.1]))
        igd_ind = IGD(true_pf_norm)
        
        ideal = true_pf_norm.min(axis=0)
        nadir = true_pf_norm.max(axis=0)
        
        # 2. Compute metrics for Baselines
        baseline_stats = {}
        for algo in ['FF', 'BFD-V', 'BFD-P']:
            if ds in baseline_fronts and algo in baseline_fronts[ds]:
                front = baseline_fronts[ds][algo]
                front_norm = (front - f_min) / denom
                hv = hv_ind(front_norm)
                igd = igd_ind(front_norm)
                
                com = front[0, 0]
                casc = -front[0, 1]
                
                baseline_stats[algo] = {
                    'dataset': ds,
                    'algorithm': algo,
                    'seed': 0,
                    'HV': hv,
                    'IGD': igd,
                    'Spread': 0.0,
                    'Spacing': 0.0,
                    'best_com': com,
                    'best_casc': casc
                }
                all_metrics.append(baseline_stats[algo])
        
        # 3. Compute metrics for NSGA-II
        nsga2_hvs = []
        nsga2_igds = []
        for seed, front in nsga2_fronts.items():
            front_norm = (front - f_min) / denom
            hv = hv_ind(front_norm)
            igd = igd_ind(front_norm)
            spread = calc_spread(front_norm, ideal, nadir)
            spacing = calc_spacing(front_norm)
            
            best_com = np.min(front[:, 0])
            best_casc = np.max(-front[:, 1])
            
            nsga2_hvs.append(hv)
            nsga2_igds.append(igd)
            
            all_metrics.append({
                'dataset': ds,
                'algorithm': 'NSGA-II',
                'seed': seed,
                'HV': hv,
                'IGD': igd,
                'Spread': spread,
                'Spacing': spacing,
                'best_com': best_com,
                'best_casc': best_casc
            })
            
        # 4. Statistical Testing vs Best Baseline (for HV)
        if len(nsga2_hvs) > 0:
            best_base_hv = max([baseline_stats[a]['HV'] for a in baseline_stats])
            base_hvs = [best_base_hv] * len(nsga2_hvs)
            
            try:
                stat, p_val = stats.wilcoxon(nsga2_hvs, base_hvs)
            except Exception:
                p_val = 1.0
                
            c_delta = cliffs_delta(nsga2_hvs, base_hvs)
            
            all_metrics.append({
                'dataset': ds,
                'algorithm': 'STAT_TEST',
                'seed': 'N/A',
                'p_value_hv': p_val,
                'cliffs_delta_hv': c_delta
            })

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(tables_dir, 'all_metrics.csv'), index=False)
    
    # Aggregation
    agg_df = metrics_df[metrics_df['algorithm'] != 'STAT_TEST'].groupby(['dataset', 'algorithm']).agg(
        HV_mean=('HV', 'mean'),
        HV_std=('HV', 'std'),
        HV_best=('HV', 'max'),
        HV_worst=('HV', 'min'),
        HV_median=('HV', 'median'),
        IGD_mean=('IGD', 'mean'),
        IGD_std=('IGD', 'std'),
        CoM_mean=('best_com', 'mean'),
        CoM_std=('best_com', 'std'),
        CASC_mean=('best_casc', 'mean'),
        CASC_std=('best_casc', 'std')
    ).reset_index()
    
    agg_df.to_csv(os.path.join(tables_dir, 'aggregated_metrics.csv'), index=False)
    
    print("Phase 6 Statistical Analysis complete. Data saved to tables/")

if __name__ == '__main__':
    main()
