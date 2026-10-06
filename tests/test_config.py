# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

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
        teacher = cfg.watermark.config.name
        assert {"kgw": "kgw", "kgwnt": "neartie", "sweet": "sweet", "kgwtk": "topk", "morphmark": "morphmark",
                "sweetnt": "sweet+neartie", "morphmarknt": "morphmark+neartie"}[fam] == teacher


def test_designed_grid_covers_the_frontier_cells():
    designed = [f for f in glob.glob(os.path.join(ROOT, "configs", "*", "*.yaml"))
                if is_designed(os.path.basename(f)[:-5])]
    assert len(designed) == 51 * 4          # 226 configs; the rest are extra MorphMark settings and the near-tie ablation cells


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("base_model: x\nwatermark:\n  type: kgw\n  config: {gamma: 0.25, not_an_option: true}\n")
    with pytest.raises(Exception):
        MainConfiguration.from_yaml(str(p))


def test_flags_map_to_teacher_classes():
    """The YAML flags of KGWWatermarkConfiguration build the classes its docstring names."""
    from robust_fp.config import KGWWatermarkConfiguration
    from robust_fp.watermarks import (KGW, DenseSelector, EntropySelector, GreenList, MorphMark, NearTieSelector,
                                      NearTieWrapper, TopKSelector)

    class Tok:
        eos_token_id = bos_token_id = pad_token_id = unk_token_id = None

        def get_vocab(self):
            return {f"t{i}": i for i in range(64)}

    greenlist = GreenList(Tok().get_vocab(), gamma=0.25, device="cpu")
    build = lambda **kw: KGWWatermarkConfiguration(**kw).build(Tok(), device="cpu", greenlist=greenlist)
    assert isinstance(build().selector, DenseSelector)
    assert isinstance(build(tie_restricted=True, tau_gap=1.0).selector, NearTieSelector)
    assert isinstance(build(entropy_restricted=True, entropy_threshold=0.5).selector, EntropySelector)
    assert isinstance(build(topk_restricted=5).selector, TopKSelector)
    mm = build(morphmark=True, morphmark_config={"variant": "log", "k_log": 1.0})
    assert isinstance(mm, MorphMark) and mm.variant == "log" and mm.k_log == 1.0
    wrapped = build(entropy_restricted=True, neartie_wrap=True, tau_gap=1.0)
    assert isinstance(wrapped, NearTieWrapper) and isinstance(wrapped.base, KGW)
    for kw in ({"tie_restricted": True}, {"tie_restricted": True, "entropy_restricted": True, "tau_gap": 1.0},
               {"neartie_wrap": True, "tau_gap": 1.0}, {"neartie_wrap": True, "tie_restricted": True, "tau_gap": 1.0},
               {"morphmark_config": {"variant": "exp"}}, {"morphmark": True, "morphmark_config": {"variant": "x"}},
               {"morphmark": True, "morphmark_config": {"k_exp": 1.0, "unknown": 1}}, {"topk_restricted": -1}):
        with pytest.raises(Exception):
            KGWWatermarkConfiguration(**kw)
    for kw, name in (({}, "kgw"), ({"tie_restricted": True, "tau_gap": 1.0}, "neartie"), ({"morphmark": True}, "morphmark"),
                     ({"morphmark": True, "neartie_wrap": True, "tau_gap": 1.0}, "morphmark+neartie")):
        assert KGWWatermarkConfiguration(**kw).name == name
