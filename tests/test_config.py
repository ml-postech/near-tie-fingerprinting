import glob
import os

import pytest

from robust_fp.config import MainConfiguration
from robust_fp.results.cells import family_of, is_designed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_every_shipped_config_parses_and_names_its_cell():
    files = sorted(glob.glob(os.path.join(ROOT, "configs", "*", "*.yaml")))
    assert len(files) == 226
    for f in files:
        cfg = MainConfiguration.from_yaml(f)
        cell = os.path.basename(f)[:-5]
        assert cfg.finetuning.custom_name == cell
        fam, _ = family_of(cell)
        gate = cfg.watermark.config.gate_name
        assert {"kgw": "kgw", "kgwnt": "neartie", "sweet": "sweet", "kgwtk": "topk", "morphmark": "morphmark",
                "sweetnt": "sweet+neartie", "morphmarknt": "morphmark+neartie"}[fam] == gate


def test_designed_grid_covers_the_frontier_cells():
    designed = [f for f in glob.glob(os.path.join(ROOT, "configs", "*", "*.yaml"))
                if is_designed(os.path.basename(f)[:-5])]
    assert len(designed) == 51 * 4          # 226 configs; the rest are extra MorphMark settings and the near-tie ablation cells


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("base_model: x\nwatermark:\n  type: kgw\n  config: {gamma: 0.25, not_an_option: true}\n")
    with pytest.raises(Exception):
        MainConfiguration.from_yaml(str(p))
