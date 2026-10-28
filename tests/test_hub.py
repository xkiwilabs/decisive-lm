"""Machine-local Hugging Face settings (configs/local.yaml, gitignored): keep weights out of a cache that a
serving stack owns."""
import pathlib

from decisivelm.model import hub_kwargs


def test_no_local_config_changes_nothing(tmp_path):
    assert hub_kwargs(tmp_path / "local.yaml") == {}


def test_local_config_sets_the_cache_and_offline_mode(tmp_path):
    cfg = tmp_path / "local.yaml"
    cfg.write_text("hf_cache: ~/models/decisive-lm/hub\noffline: true\n")
    assert hub_kwargs(cfg) == {"cache_dir": str(pathlib.Path.home() / "models/decisive-lm/hub"),
                               "local_files_only": True}


def test_offline_can_be_dropped_for_dataset_downloads(tmp_path):
    cfg = tmp_path / "local.yaml"
    cfg.write_text("hf_cache: /data/hub\noffline: true\n")
    assert hub_kwargs(cfg, offline=False) == {"cache_dir": "/data/hub"}
