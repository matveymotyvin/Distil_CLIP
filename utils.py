# Import the pathlib module to handle file paths
from pathlib import Path

# Define a function to load a model path based on given arguments
def load_model_path(root=None, version=None, v_num=None, best=False):
    # Define a helper function to sort checkpoint paths by epoch number
    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    # Define a helper function to generate the root path
    def generate_root():
        # Check if root path is specified, else check for version number or version name
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        elif v_num is not None:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))
        else:
            return None

    # Generate the root path
    root = generate_root()
    # Check if the root path is valid
    if root is None or not Path(root).is_dir():
        return None

    # If best=True, find the checkpoint with the highest epoch number
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        if not files:
            return None
        # Sort the checkpoint paths by epoch number
        files.sort(key=sort_by_epoch, reverse=True)
        return str(files[0])
    # Else, return the path to the last checkpoint
    else:
        return str(Path(root) / 'last.ckpt')

# Define a function to load a model path based on parsed command-line arguments
def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num, best=args.load_best)
