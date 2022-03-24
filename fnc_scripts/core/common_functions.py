"""Common functions used in COINSTAC scripts.
"""


def list_recursive(d, key):
    """Yields the value corresponding to key in a dict d."""
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v
