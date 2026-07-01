import pytest

from micro_sam.sam_annotator._state import AnnotatorState, Singleton


@pytest.fixture(autouse=True)
def reset_annotator_state():
    # 'AnnotatorState' is a process-wide singleton, so state set by one test leaks into the next and
    # makes the GUI tests order-dependent. Drop the cached instance around each test so every test
    # starts from a fresh state, with all dataclass fields back to their defaults.
    Singleton._instances.pop(AnnotatorState, None)
    yield
    Singleton._instances.pop(AnnotatorState, None)
