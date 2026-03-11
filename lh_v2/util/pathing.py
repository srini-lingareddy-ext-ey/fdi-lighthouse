import pathlib as pth


def get_root_dir() -> pth.Path:
    return pth.Path(__file__).parent.parent.parent


def get_output_dir(sub_dir_name: str, base_dir: pth.Path | None = None) -> pth.Path:
    """
    Get the path to the output directory based on the provided base directory and subdirectory name.

    Parameters
    ----------
    sub_dir_name : str
        The name of the subdirectory to get.
    base_dir : pth.Path | None
        The base directory under which the subdirectory is located. If None, the default root directory is used.

    Returns
    -------
    pth.Path
        The path to the specified subdirectory.
    """
    if base_dir is None:
        base_dir = get_root_dir() / 'saves'
    output_dir = base_dir / sub_dir_name
    return output_dir


def create_output_dir(sub_dir_name: str, base_dir: pth.Path | None = None) -> pth.Path:
    """
    Creates a new directory for output files based on the provided base directory and subdirectory name.

    Parameters
    ----------
    sub_dir_name : str
        The name of the subdirectory to create.
    base_dir : pth.Path | None
        The base directory under which the new subdirectory will be created. If None, the default root directory is used.

    Returns
    -------
    pth.Path
        The path to the newly created subdirectory.
    """
    output_dir = get_output_dir(sub_dir_name=sub_dir_name, base_dir=base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
