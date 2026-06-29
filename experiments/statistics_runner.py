import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def cliffs_delta(lst1, lst2):
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0:
        return 0.0
    mat = np.zeros((m, n))
    for i, x in enumerate(lst1):
        for j, y in enumerate(lst2):
            if x > y: mat[i,j] = 1
            elif x < y: mat[i,j] = -1
    return np.sum(mat) / (m * n)

def compute_statistics():
    base_dir = os.path.dirname(__file__)
    results_path = os.path.join(base_dir, 'results.csv')
    
    if not os.path.exists(results_path):
        print("Results file not found. Ensure Phase 5 is completed.")
        return
        
    df = pd.read_csv(results_path)
    
    # 1. Summary Statistics
    summary = df.groupby(['dataset', 'algorithm']).agg(
        com_mean=('com_deviation', 'mean'),
        com_median=('com_deviation', 'median'),
        com_std=('com_deviation', 'std'),
        casc_mean=('casc_score', 'mean'),
        casc_median=('casc_score', 'median'),
        casc_std=('casc_score', 'std'),
        util_mean=('utilization_pct', 'mean'),
        runtime_mean=('runtime_s', 'mean')
    ).reset_index()
    
    # Calculate 95% CI for NSGA-II
    nsga2_runs = df[df['algorithm'] == 'NSGA-II']
    ci_data = []
    for ds in df['dataset'].unique():
        ds_data = nsga2_runs[nsga2_runs['dataset'] == ds]
        if len(ds_data) > 1:
            com_ci = stats.t.interval(0.95, len(ds_data)-1, loc=np.mean(ds_data['com_deviation']), scale=stats.sem(ds_data['com_deviation']))
            casc_ci = stats.t.interval(0.95, len(ds_data)-1, loc=np.mean(ds_data['casc_score']), scale=stats.sem(ds_data['casc_score']))
            ci_data.append({
                'dataset': ds, 'algorithm': 'NSGA-II',
                'com_ci_low': com_ci[0], 'com_ci_high': com_ci[1],
                'casc_ci_low': casc_ci[0], 'casc_ci_high': casc_ci[1]
            })
    
    if ci_data:
        ci_df = pd.DataFrame(ci_data)
        summary = pd.merge(summary, ci_df, on=['dataset', 'algorithm'], how='left')
    
    summary.to_csv(os.path.join(base_dir, 'summary_statistics.csv'), index=False)
    
    # Generate LaTeX table
    latex_str = summary.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(base_dir, 'summary_statistics.tex'), 'w') as f:
        f.write(latex_str)
        
    # 2. Wilcoxon & Cliff's Delta (Paired over datasets)
    # We compare Median NSGA-II against FF, BFD-V, BFD-P for each dataset
    metrics_data = []
    nsga2_medians = summary[summary['algorithm'] == 'NSGA-II'].set_index('dataset')
    
    for baseline in ['FF', 'BFD-V', 'BFD-P']:
        base_df = summary[summary['algorithm'] == baseline].set_index('dataset')
        
        # Ensure we only compare matching datasets
        common_idx = nsga2_medians.index.intersection(base_df.index)
        if len(common_idx) == 0:
            continue
            
        nsga_com = nsga2_medians.loc[common_idx, 'com_median'].values
        base_com = base_df.loc[common_idx, 'com_median'].values
        
        nsga_casc = nsga2_medians.loc[common_idx, 'casc_median'].values
        base_casc = base_df.loc[common_idx, 'casc_median'].values
        
        # Wilcoxon requires differences
        try:
            w_com, p_com = stats.wilcoxon(nsga_com, base_com)
        except:
            w_com, p_com = float('nan'), float('nan')
            
        try:
            w_casc, p_casc = stats.wilcoxon(nsga_casc, base_casc)
        except:
            w_casc, p_casc = float('nan'), float('nan')
            
        c_com = cliffs_delta(nsga_com, base_com)
        c_casc = cliffs_delta(nsga_casc, base_casc)
        
        metrics_data.append({
            'Comparison': f'NSGA-II vs {baseline}',
            'CoM_Wilcoxon_p': p_com,
            'CoM_Cliffs_Delta': c_com,
            'CASC_Wilcoxon_p': p_casc,
            'CASC_Cliffs_Delta': c_casc
        })
        
    tests_df = pd.DataFrame(metrics_data)
    tests_df.to_csv(os.path.join(base_dir, 'statistical_tests.csv'), index=False)
    with open(os.path.join(base_dir, 'statistical_tests.tex'), 'w') as f:
        f.write(tests_df.to_latex(index=False, float_format="%.4e"))
        
    # 3. Hypervolume and IGD calculation
    # We must construct a Reference Pareto Front for each dataset by combining all 30 runs
    pareto_dir = os.path.join(base_dir, 'pareto_fronts')
    hv_igd_data = []
    
    for ds in df['dataset'].unique():
        # Combine all fronts for this dataset
        all_F = []
        ds_runs = df[(df['algorithm'] == 'NSGA-II') & (df['dataset'] == ds)]
        
        for seed in ds_runs['seed']:
            pf_path = os.path.join(pareto_dir, f"{ds}_seed_{seed}.csv")
            if os.path.exists(pf_path):
                pf = pd.read_csv(pf_path).values
                all_F.append(pf)
                
        if not all_F:
            continue
            
        combined_F = np.vstack(all_F)
        # Find non-dominated points among all combined points
        nds = NonDominatedSorting().do(combined_F, only_non_dominated_front=True)
        ref_F = combined_F[nds]
        
        # Calculate max bounds for HV
        ref_point = ref_F.max(axis=0) * 1.1 
        
        hv = HV(ref_point=ref_point)
        igd = IGD(ref_F)
        
        for idx, pf in enumerate(all_F):
            try:
                run_hv = hv.do(pf)
                run_igd = igd.do(pf)
                hv_igd_data.append({
                    'dataset': ds,
                    'seed': ds_runs['seed'].iloc[idx],
                    'hypervolume': run_hv,
                    'igd': run_igd
                })
            except:
                pass
                
    if hv_igd_data:
        hv_df = pd.DataFrame(hv_igd_data)
        hv_summary = hv_df.groupby('dataset').agg(
            hv_mean=('hypervolume', 'mean'),
            hv_std=('hypervolume', 'std'),
            igd_mean=('igd', 'mean'),
            igd_std=('igd', 'std')
        ).reset_index()
        
        hv_summary.to_csv(os.path.join(base_dir, 'pareto_metrics.csv'), index=False)
        with open(os.path.join(base_dir, 'pareto_metrics.tex'), 'w') as f:
            f.write(hv_summary.to_latex(index=False, float_format="%.4f"))

    print("Phase 6 Statistical Analysis completed.")

if __name__ == "__main__":
    compute_statistics()
