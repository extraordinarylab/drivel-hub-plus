import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_inference.py"
SPEC = importlib.util.spec_from_file_location("run_inference", MODULE_PATH)
run_inference = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_inference)


def test_final_answer_removes_leaked_reasoning() -> None:
    assert run_inference.final_answer("analysis\n</think>\n答案") == "答案"


def test_final_answer_preserves_plain_response() -> None:
    assert run_inference.final_answer("  plain answer  ") == "plain answer"
