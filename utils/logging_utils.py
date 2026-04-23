"""Lightweight verbose-aware print utility for Book2Audio."""

from __future__ import annotations


def vprint(verbose: bool, *args: object, **kwargs: object) -> None:
    """Print only if verbose is True.

    Drop-in replacement for print that respects a verbose flag. Pass the
    caller's verbose flag as the first argument; all remaining arguments
    are forwarded to print unchanged.

    Args:
        verbose: If False, nothing is printed.
        *args: Positional arguments forwarded to print.
        **kwargs: Keyword arguments forwarded to print.
    """
    if verbose:
        print(*args, **kwargs)
