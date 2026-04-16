"""
Seed Manager - Global seed management for reproducibility
"""

# Global variable to store the random seed value
_RANDOM_SEED = None


def set_random_seed(seed_value):
    """
    Set the global random seed value.

    Args:
        seed_value: The seed value (int). If 0, seeding is disabled.
    """
    global _RANDOM_SEED
    _RANDOM_SEED = seed_value


def get_random_state():
    """
    Get the random_state parameter for sklearn functions.

    Returns:
        int: The seed value if seeding is enabled (> 0), None otherwise
    """
    if _RANDOM_SEED is None or _RANDOM_SEED == 0:
        return None
    return _RANDOM_SEED

