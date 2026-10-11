"""Select this repository's DAM4SAM fork, never another installed SAM2 fork."""
from pathlib import Path
import sys


def activate():
    root = Path(__file__).resolve().parents[1] / "third_party" / "dam4sam"
    expected = root / "sam2" / "__init__.py"
    if not expected.is_file():
        raise ImportError(f"Bundled DAM4SAM core is missing: {expected}")
    existing = sys.modules.get("sam2")
    if existing is not None and Path(existing.__file__).resolve() != expected.resolve():
        raise ImportError("A different SAM2 fork is already imported; start a fresh Python process")
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    return root
