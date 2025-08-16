import pandas as pd
import numpy as np
import os

def get_data(mode, model_config, data_args):
    """
    Defines the paths for the local and global features. Provides the reports and abnormality labels.

    Args:
        mode: Defines whether the data loaded is for training ("train") or validation ("valid").
        model_config: Contains model-specific parameters based on report_gen/config.py 
        data_args: Contains data-specific parameters based on report_gen/config.py 
        use_s3:

    Returns:
        local_paths:  list[str]   # per-case path to local features 
        global_paths: list[str]   # per-case path to global features 
        reports:      list[str]   # per-case report text
        labels:       np.ndarray  # abnormality labels of shape [N, C]

    Update the local/global feature file suffixes below to match your saved files.
    """
    
    # Filter samlpes based on duplicates of original reports from CT-RATE
    reports_path = os.path.join(data_args.ctrate_data_path, f'{mode}_reports.csv')
    ct_rate_df = pd.read_csv(reports_path).drop_duplicates('Findings_EN')
    names = list(ct_rate_df['VolumeName'])   
    
    # Take the reports (full text) from Radgenome dataset for actual training/validation 
    region_reports_path = os.path.join(data_args.ctrate_data_path, f'{mode}_region_reports.csv')
    df_rr=pd.read_csv(region_reports_path)
    radgenome_df=df_rr[df_rr['Anatomy'].isna()].sort_values('Volumename')
    filtered_df = radgenome_df[radgenome_df['Volumename'].isin(names)]
    filtered_df['Volumename'] = pd.Categorical(filtered_df['Volumename'], categories=names, ordered=True)
    filtered_df = filtered_df.sort_values('Volumename')
    reports = list(filtered_df['Sentence'].values)
    
    # Define the exact paths for local and global features. Adapt paths accordingly. 
    ct_paths = [f'dataset/{mode}/{name[:-11]}/{name[:-9]}/{name}' for name in names]
    final_layer = len(model_config.feat_dims)-1
    local_paths= [elem.replace('dataset', f'features/{model_config.arch}/downsampling/ip/layer_0').replace('.nii.gz',f'.npy') for elem in ct_paths]
    global_paths = [elem.replace('dataset', f'features/{model_config.arch}/downsampling/ip/layer_{final_layer}').replace('.nii.gz',f'_full.npy') for elem in ct_paths]
    
    # If S3 is not used, features are assumed to be located at following paths: 
    if not data_args.use_s3:
        local_paths = [os.path.join(data_args.ctrate_data_path,path) for path in local_paths]
        global_paths = [os.path.join(data_args.ctrate_data_path,path) for path in global_paths]
        
    if mode == 'valid':
        abnormality_path = os.path.join(data_args.ctrate_data_path, f'multi_abnormality_labels_{mode}.csv')
        abnormality_df=pd.read_csv(abnormality_path)
        abnormality_df = abnormality_df[abnormality_df['VolumeName'].isin(ct_rate_df['VolumeName'])]
        abnormality_labels = np.column_stack([abnormality_df[col].values for col in abnormality_df.columns if col != 'VolumeName'])
    
        return local_paths, global_paths, reports, abnormality_labels
    
    return local_paths, global_paths, reports