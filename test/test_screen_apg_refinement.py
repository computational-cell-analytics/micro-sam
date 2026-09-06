import importlib.util
import sys
from pathlib import Path


_EVALUATION_DIR = Path(__file__).parents[1] / "finetuning/v2/evaluation"
sys.path.insert(0, str(_EVALUATION_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "screen_apg_refinement", _EVALUATION_DIR / "optimization" / "screen_apg_refinement.py",
)
screen_apg_refinement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(screen_apg_refinement)


class _Gate:
    def __init__(self, stage):
        self.gate_stage = stage


def test_postmerge_gate_is_not_scored_during_proposal_generation():
    assert screen_apg_refinement._compute_premerge_gate_scores(True, False, _Gate("premerge"))
    assert not screen_apg_refinement._compute_premerge_gate_scores(True, False, _Gate("postmerge"))
    assert not screen_apg_refinement._compute_premerge_gate_scores(True, True, _Gate("premerge"))
