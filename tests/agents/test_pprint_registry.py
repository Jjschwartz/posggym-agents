"""Tests that `posggym_agents.pprint_registry` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_pprint_registry.py
"""
import posggym_agents as pga


# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa

reduced_registry = {pi_id: pi_spec for pi_id, pi_spec in pga.registry.items()}


def test_pprint_custom_registry():
    """Testing a registry different from default."""
    a = {
        "Random-v0": pga.registry["Random-v0"],
        "LevelBasedForaging-5x5-n2-f4-v2/Heuristic1-v0": pga.registry[
            "LevelBasedForaging-5x5-n2-f4-v2/Heuristic1-v0"
        ],
    }
    out = pga.pprint_registry(a, disable_print=True)

    correct_out = """===== Generic =====
Random-v0

===== LevelBasedForaging-5x5-n2-f4-v2 =====
Heuristic1-v0

"""
    assert out == correct_out
