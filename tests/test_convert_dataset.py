"""scripts/convert_dataset.py --converter: a user's own converter, loaded from its file."""
import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location("convert_dataset",
                                              pathlib.Path(__file__).parent.parent / "scripts/convert_dataset.py")
convert_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_dataset)


def test_user_converter_is_loaded_from_its_file(tmp_path):
    (tmp_path / "mine.py").write_text("def convert(source_dir):\n    return [str(source_dir)]\n")
    assert convert_dataset.user_converter(f"{tmp_path}/mine.py:convert")("raw") == ["raw"]


@pytest.mark.parametrize("spec, message", [("mine.py", "FILE.py:FUNCTION"), ("{d}/nothere.py:convert", "no Python"),
                                           ("{d}/mine.py:missing", "no function")])
def test_user_converter_errors_say_what_is_wrong(tmp_path, spec, message):
    (tmp_path / "mine.py").write_text("def convert(source_dir):\n    return []\n")
    with pytest.raises(SystemExit, match=message):
        convert_dataset.user_converter(spec.format(d=tmp_path))
