"""Resolve saved expanded-evidence paths after moving the project folder."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def local_path(value):
    path=Path(value)
    parts=path.parts
    for i in range(len(parts)-1):
        if parts[i:i+2]==('artifacts','expanded'):
            return ROOT/Path(*parts[i:])
    return path if path.is_absolute() else ROOT/path


def comparable_task(task):
    return {key:(str(local_path(value).relative_to(ROOT)) if key in ['cache_path','folder'] else value) for key,value in task.items()}
