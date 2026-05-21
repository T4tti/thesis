import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def create_charts():
    # Read the data
    file_path = 'e:/thesis/result.xlsx'
    df = pd.read_excel(file_path)
    
    # Rename the first column to 'Model'
    df.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)
    
    # Ensure the output directory exists
    output_dir = 'e:/thesis/artifacts/models'
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['Accuracy', 'F1', 'Recall', 'Precision']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'] # basic seaborn palette
    
    sns.set(style="whitegrid", rc={"axes.titlesize": 14, "axes.labelsize": 12})
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        
        # Sort values descending for better visualization, or keep original order. Let's keep original or sort. 
        # Usually, keeping original order if it's meaningful, but sorting by the metric highlights the best.
        # Let's keep the original order to compare models consistently across plots.
        
        ax = sns.barplot(x='Model', y=metric, data=df, palette='viridis')
        
        plt.title(f'{metric} Comparison across Models')
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.ylim(0.8, 1.0) # The values are around 0.85-0.92, so zoom in to see differences
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.4f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{metric.lower()}_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {metric} chart to {save_path}")

if __name__ == '__main__':
    create_charts()
