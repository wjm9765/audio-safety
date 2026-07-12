"""Load a project ``.env`` into ``os.environ`` via python-dotenv.

Secrets such as ``OPENROUTER_API_KEY`` live in ``.env`` (git-ignored) on the GPU
run environment instead of being exported by hand. ``load_dotenv`` walks up from
the current directory to find ``.env``, is a no-op when the file is absent (e.g.
CPU tests / CI), and never clobbers a variable already set in the environment
unless ``override=True``. The import is done lazily so importing this module (or
the config loader) never hard-fails if the dependency is somehow missing.
"""

_LOADED = False


def load_project_dotenv(*, override: bool = False) -> bool:
    """Populate ``os.environ`` from the nearest ``.env``. Returns True if loaded.

    Idempotent: only the first call reads the file (pass ``override=True`` to
    force a reload). Existing environment variables win unless ``override``.
    """
    global _LOADED
    if _LOADED and not override:
        return False
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return False
    _LOADED = True
    # usecwd=True walks up from the current working directory (where runs are
    # launched) rather than from this file's location, so the project .env is
    # found on the GPU checkout and unrelated ancestor .env files are not.
    path = find_dotenv(usecwd=True)
    if not path:
        return False
    return bool(load_dotenv(path, override=override))
