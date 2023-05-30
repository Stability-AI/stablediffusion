import os


def get_configs_path() -> str:
    """
    Get the `configs` directory.

    For a working copy, this is the one in the root of the repository,
    but for an installed copy, it's in the `ldm` package (see pyproject.toml).
    """
    this_dir = os.path.dirname(__file__)
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find LDM configs in {candidates}")
