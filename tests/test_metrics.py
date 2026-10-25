import ast
from pathlib import Path

# Load only the pure metric definitions without importing torch/transformers.
source = (Path(__file__).resolve().parents[1] / "experiment.py").read_text()
module = ast.parse(source)
fns = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in {"topk_overlap", "relative_restoration"}]
namespace = {}
exec(compile(ast.Module(body=fns, type_ignores=[]), "<metrics>", "exec"), namespace)


def test_overlap():
    fn = namespace['topk_overlap']
    assert fn([1, 5, 2], [1, 5, 2], 2) == 1
    assert fn([1, 5, 2], [5, 1, 2], 1) == 0
    assert fn([], [], 2) is None


def test_restoration():
    fn = namespace['relative_restoration']
    assert fn(2, 0, 1) == 0.5
    assert fn(0, 0, 0) is None
    assert fn(2, 0, 0) == 0
