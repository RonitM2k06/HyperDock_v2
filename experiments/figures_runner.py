import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IEEE Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 600,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

def save_fig(fig, path):
    fig.savefig(path + '.png', format='png')
    fig.savefig(path + '.svg', format='svg')
    fig.savefig(path + '.pdf', format='pdf')
    plt.close(fig)

def main():
    exp_dir = os.path.dirname(__file__)
    res_path = os.path.join(exp_dir, 'results.csv')
    conv_path = os.path.join(exp_dir, 'convergence.csv')
    metrics_path = os.path.join(exp_dir, 'tables', 'aggregated_metrics.csv')
    pareto_dir = os.path.join(exp_dir, 'pareto_fronts')
    fig_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(res_path): return
    df = pd.read_csv(res_path)
    conv_df = pd.read_csv(conv_path) if os.path.exists(conv_path) else None
    
    datasets = df['dataset'].unique()
    
    # 1. Pareto Fronts (CASC vs CoM Trade-off)
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(5, 4))
        ds_df = df[df['dataset'] == ds]
        
        # Plot baselines
        colors = {'FF': 'blue', 'BFD-V': 'green', 'BFD-P': 'orange'}
        markers = {'FF': 's', 'BFD-V': '^', 'BFD-P': 'D'}
        
        for algo in ['FF', 'BFD-V', 'BFD-P']:
            b_df = ds_df[ds_df['algorithm'] == algo]
            if not b_df.empty:
                ax.scatter(b_df['com_deviation'].values[0], b_df['casc_score'].values[0], 
                           color=colors[algo], marker=markers[algo], s=50, label=algo, zorder=5)
                           
        # Plot NSGA-II Fronts (scatter of all 30 seeds)
        nsga2_x, nsga2_y = [], []
        nsga2_runs = ds_df[ds_df['algorithm'] == 'NSGA-II']
        for seed in nsga2_runs['seed']:
            pf_file = os.path.join(pareto_dir, f"{ds}_seed_{seed}.csv")
            if os.path.exists(pf_file):
                pf = pd.read_csv(pf_file).values
                nsga2_x.extend(pf[:, 0])
                nsga2_y.extend(-pf[:, 1]) # Convert neg_casc back to casc
                
        if nsga2_x:
            ax.scatter(nsga2_x, nsga2_y, color='red', alpha=0.3, s=10, label='NSGA-II Solutions', zorder=1)
            
        ax.set_xlabel('Center of Mass Deviation (m) $\downarrow$')
        ax.set_ylabel('CASC Score $\\uparrow$')
        ax.set_title(f'Pareto Fronts: {ds}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        save_fig(fig, os.path.join(fig_dir, f'pareto_front_{ds}'))

    # 4. Runtime Scaling
    fig, ax = plt.subplots(figsize=(5, 4))
    sizes = []
    runtimes = {'FF': [], 'BFD-V': [], 'BFD-P': [], 'NSGA-II': []}
    
    # Extract N from dataset names (e.g. HB-V1-SC1-100 -> 100)
    agg_df = df.groupby(['dataset', 'algorithm'])['runtime_s'].mean().reset_index()
    for ds in datasets:
        n = int(ds.split('-')[-1])
        if n not in sizes: sizes.append(n)
        
    sizes.sort()
    for algo in runtimes.keys():
        means = []
        for n in sizes:
            # find dataset with this N
            ds_name = next(ds for ds in datasets if ds.endswith(str(n)))
            val = agg_df[(agg_df['dataset'] == ds_name) & (agg_df['algorithm'] == algo)]['runtime_s']
            means.append(val.values[0] if not val.empty else np.nan)
        ax.plot(sizes, means, marker='o', label=algo)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Items (N)')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime Scalability')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    save_fig(fig, os.path.join(fig_dir, 'runtime_scaling'))
    
    # 5. Convergence Curves & 7. Constraint Violation Curves
    if conv_df is not None:
        for ds in datasets:
            # Filter runs for this dataset
            run_ids = [r for r in conv_df['run_id'].unique() if r.startswith(ds)]
            ds_conv = conv_df[conv_df['run_id'].isin(run_ids)]
            
            if ds_conv.empty: continue
            
            # Group by generation
            gen_agg = ds_conv.groupby('generation').agg({
                'best_com_dev': ['mean', 'std'],
                'best_neg_casc': ['mean', 'std'],
                'cv_min': ['mean', 'std']
            })
            
            gens = gen_agg.index.values
            
            # Objective Convergence
            fig, ax1 = plt.subplots(figsize=(5, 4))
            color = 'tab:blue'
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('CoM Deviation (m)', color=color)
            mean_com = gen_agg['best_com_dev']['mean'].values
            std_com = gen_agg['best_com_dev']['std'].values
            ax1.plot(gens, mean_com, color=color, label='CoM Dev')
            ax1.fill_between(gens, mean_com-std_com, mean_com+std_com, alpha=0.2, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('CASC Score', color=color)
            mean_casc = -gen_agg['best_neg_casc']['mean'].values
            std_casc = gen_agg['best_neg_casc']['std'].values
            ax2.plot(gens, mean_casc, color=color, label='CASC')
            ax2.fill_between(gens, mean_casc-std_casc, mean_casc+std_casc, alpha=0.2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            plt.title(f'Convergence: {ds}')
            save_fig(fig, os.path.join(fig_dir, f'convergence_{ds}'))
            
            # Constraint Violation
            fig, ax = plt.subplots(figsize=(5, 4))
            mean_cv = gen_agg['cv_min']['mean'].values
            std_cv = gen_agg['cv_min']['std'].values
            ax.plot(gens, mean_cv, color='black', label='CV')
            ax.fill_between(gens, np.maximum(0, mean_cv-std_cv), mean_cv+std_cv, alpha=0.2, color='gray')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Constraint Violation (Unplaced Vol/Count)')
            ax.set_title(f'Constraint Violation: {ds}')
            ax.grid(True, ls='--', alpha=0.5)
            save_fig(fig, os.path.join(fig_dir, f'constraint_violation_{ds}'))

    # 6. Boxplots
    if os.path.exists(metrics_path):
        mdf = pd.read_csv(os.path.join(exp_dir, 'tables', 'all_metrics.csv'))
        mdf = mdf[mdf['algorithm'] != 'STAT_TEST']
        
        # HV Boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        data_to_plot = []
        labels = []
        for ds in datasets:
            nsga2_hv = mdf[(mdf['dataset'] == ds) & (mdf['algorithm'] == 'NSGA-II')]['HV'].dropna().values
            if len(nsga2_hv) > 0:
                data_to_plot.append(nsga2_hv)
                labels.append(ds.split('-')[-1])
                
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_xlabel('Dataset Size (N)')
            ax.set_ylabel('Hypervolume (HV)')
            ax.set_title('NSGA-II Hypervolume Distribution')
            ax.grid(True, ls='--', alpha=0.5)
            save_fig(fig, os.path.join(fig_dir, 'boxplot_hv'))

    print("Phase 8 Figure Generation complete. Plots saved to figures/")

if __name__ == '__main__':
    main()
