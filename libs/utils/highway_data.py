
from pathlib import Path

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def parse_data_roots_arg(values):
    if not values:
        return None
    roots = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'data root must be KEY=PATH, got: {value}')
        key, path = value.split('=', 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            raise ValueError(f'data root must be KEY=PATH, got: {value}')
        if key in roots:
            raise ValueError(f'duplicate data root key: {key}')
        roots[key] = path
    return roots


def normalize_data_roots(data_root=None, data_roots=None):
    if data_roots:
        if isinstance(data_roots, dict):
            items = data_roots.items()
        elif isinstance(data_roots, (list, tuple)):
            parsed = parse_data_roots_arg(data_roots)
            items = parsed.items()
        else:
            raise TypeError(f'Unsupported data_roots type: {type(data_roots)!r}')
        roots = {str(key): Path(path) for key, path in items}
        if not roots:
            raise ValueError('data_roots cannot be empty')
        return roots, next(iter(roots))
    if data_root is None:
        raise ValueError('Either data_root or data_roots must be provided')
    return {'default': Path(data_root)}, 'default'


def parse_split_line(line, data_roots, default_root_key):
    parts = str(line).strip().split()
    if not parts:
        return None, None
    if len(parts) >= 2 and parts[0] in data_roots:
        return parts[0], parts[1].lstrip('/')
    return default_root_key, parts[0].lstrip('/')


def resolve_image_path(data_roots, root_key, image_rel):
    if root_key not in data_roots:
        raise KeyError(f'Unknown data root key {root_key!r}; available={list(data_roots)}')
    rel = str(image_rel).strip().split()[0].lstrip('/')
    direct = Path(rel)
    if direct.is_absolute() and direct.exists():
        return direct
    data_root = data_roots[root_key]
    candidate = data_root / rel
    if candidate.exists():
        return candidate
    stem = Path(rel).stem
    matches = [data_root / f'{stem}{suffix}' for suffix in IMAGE_SUFFIXES]
    matches = [p for p in matches if p.exists()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'Image listed in split not found: root={root_key} item={image_rel}')
    raise RuntimeError(f'Ambiguous image stem root={root_key} item={image_rel}: {matches}')


def relative_to_root(path, data_root):
    return Path(path).relative_to(Path(data_root)).as_posix()


def display_image_name(root_key, image_rel, multi_root=False):
    return f'{root_key}/{image_rel}' if multi_root else image_rel
