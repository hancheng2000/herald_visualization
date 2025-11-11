import pandas as pd
from ruamel.yaml import YAML
import os, glob, shutil

def chemistry_se(energy_am,cell_id):
    spec_df = pd.read_csv('/scratch/venkvis_root/venkvis/shared_data/herald/In-house cells and syntheses - cell-design-input.csv')
    coeffs = {"70_3_30": 0.63,
    "75_3_30": 0.67,
    "80_3_30": 0.71,
    "80_5_30": 0.71,
    "85_1_30": 0.77,
    "85_2_30": 0.77,
    "85_3_30": 0.76,
    "85_5_30": 0.761,
    "87_3_30": 0.77,
    '60_15_52': 0.47,
    '80_5_52': 0.63,
    '90_5_52': 0.72,
    '70_5_52': 0.54,
    '85_5_52': 0.67,
    '90_5_30': 0.81,
    '60_15_30': 0.54,
    '70_5_30': 0.63,
    } # AM to chem coefficient
    df1 = spec_df[spec_df['cell id']==cell_id].copy()
    if len(df1)>0:
        fef3_mass = int(df1['cathode FeF3 mass fraction'].values[0]*100)
        binder_mass = int(df1['cathode binder mass fraction'].values[0]*100)
        porosity = int(df1['porosity'].values[0]*100)
        coeff = coeffs[f'{fef3_mass}_{binder_mass}_{porosity}']
    else:
        return None
    discharge_se_lst_chem = energy_am * coeff
    return discharge_se_lst_chem

if __name__ == '__main__':
    dirs = glob.glob(r'*CC[1-9][0-9][0-9][A-Z]*/', root_dir='/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/BCS905') # ignore the 0xx series
    dirs = sorted(dirs) 
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    energy_dict = {}
    chem_se_dict = {}
    sc_dict = {}
    chem_sc_dict = {}
    vavg_dict = {}
    for i, dd in enumerate(dirs):
        full_path = os.path.join('/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing/BCS905', dd)
        cell_id = [ii for ii in dd.split('_') if 'CC' in ii][0]
        cell_id = cell_id.replace('CC','')
        print(full_path)
        if os.path.exists(os.path.join(full_path, 'outputs', 'cycle_summary.csv')):
            # try:
            df = pd.read_csv(os.path.join(full_path, 'outputs', 'cycle_summary.csv'))
            if len(df) == 0:
                continue
            if 'Specific Discharge Energy Total AM' in df.columns:
                energy_dict[cell_id] = df['Specific Discharge Energy Total AM'].tolist()
                chem_se = chemistry_se(df['Specific Discharge Energy Total AM'].to_numpy(),cell_id)                
                if chem_se is None or len(chem_se) == 0:
                    chem_se_dict[cell_id] = None
                else:
                    chem_se_dict[cell_id] = chem_se.tolist()
            if 'Specific Discharge Capacity Total AM' in df.columns:
                sc_dict[cell_id] = df['Specific Discharge Capacity Total AM'].tolist()
                chem_sc = chemistry_se(df['Specific Discharge Capacity Total AM'].to_numpy(),cell_id)
                if chem_sc is None or len(chem_sc) == 0:
                    chem_sc_dict[cell_id] = None
                else:
                    chem_sc_dict[cell_id] = chem_sc.tolist()
            if ('Specific Discharge Capacity Total AM' in df.columns) and ('Specific Discharge Energy Total AM' in df.columns):
                vavg = df['Specific Discharge Energy Total AM'].to_numpy() / df['Specific Discharge Capacity Total AM'].to_numpy()
                if vavg is None or len(vavg) == 0:
                    vavg_dict[cell_id] = None
                else:
                    vavg_dict[cell_id] = vavg.tolist()
            # except:
            #     print(f"Error reading summary file in {full_path}")
        else:
            print(f"No summary file in {full_path}")
            continue
        
    with open(f'/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_se_am_dict.yaml', 'w') as f:
        yaml.dump(energy_dict, f)
    with open(f'/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_se_chem_dict.yaml', 'w') as f:
        yaml.dump(chem_se_dict, f)
    with open(f'/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_sc_am_dict.yaml', 'w') as f:
        yaml.dump(sc_dict, f)
    with open(f'/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_sc_chem_dict.yaml', 'w') as f:
        yaml.dump(chem_sc_dict, f)
    with open(f'/scratch/venkvis_root/venkvis/shared_data/herald/cell_id_energy_list/hypo_vavg_am_dict.yaml', 'w') as f:
        yaml.dump(vavg_dict, f)