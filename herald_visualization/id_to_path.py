data_path = "/scratch/venkvis_root/venkvis/shared_data/herald/Electrochemical_Testing" 
import os 
import glob

def id_to_path(cellid, root_dir=data_path):
    """
    Find the correct directory path to a data folder from the cell ID
    """

    glob_str = os.path.join('**/outputs/*'+cellid+'_*.csv')
    paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
    if len(paths) == 1:
        return os.path.join(root_dir, paths[0])
    elif len(paths) == 0: 
        glob_str = os.path.join('**/outputs/*'+cellid+'*.csv')
        paths = glob.glob(glob_str, root_dir=root_dir, recursive=True)
        if len(paths) == 1:
            return os.path.join(root_dir, paths[0])
        elif len(paths) == 0:
            print(f"No paths matched for {cellid}")
            return None
        else:
            print(f"Too many paths matched for {cellid}: {paths}")
            return [os.path.join(root_dir, path) for path in paths]
    else:
        print(f"Too many paths matched for {cellid}: {paths}")
        return [os.path.join(root_dir, path) for path in paths]