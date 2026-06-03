import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blend_sameindex_geometry.py"
spec = importlib.util.spec_from_file_location("blend_sameindex_geometry", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_read_objects_ignores_comments(tmp_path):
    p = tmp_path / "objects.txt"
    p.write_text("# comment\nshoe\n\ntrain\n")
    assert module.read_objects(p) == ["shoe", "train"]
