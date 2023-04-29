from pathlib import Path

def load_model_path(root=None, version=None, v_num=None, best=False):
    """
    Returns the path to the checkpoint file of a trained model.

    Args:
        root (str, optional): Root directory where the checkpoint file is located.
        version (str, optional): Name of the directory containing the checkpoints.
        v_num (int, optional): Numeric version of the directory containing the checkpoints.
        best (bool, optional): Whether to return the path to the best checkpoint file.

    Returns:
        str: Path to the checkpoint file.
    """
    def sort_by_epoch(path):
        """
        Returns the epoch number of a checkpoint file.
        """
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    # if no directory information is provided, return None
    if not any([root, version, v_num]):
        return None

    # if only version information is provided, construct the root path
    if not root:
        if version:
            root = Path('lightning_logs', version, 'checkpoints')
        else:
            root = Path('lightning_logs', f'version_{v_num}', 'checkpoints')

    # if root is a file, return its path
    if root.is_file():
        return str(root)

    # otherwise, list all files in the directory and sort by epoch number
    files = list(root.glob('*'))
    if best:
        files = [f for f in files if f.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)

    return str(files[0]) if files else None


def load_model_path_by_args(args):
    """
    Returns the path to the checkpoint file of a trained model using command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        str: Path to the checkpoint file.
    """
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)
