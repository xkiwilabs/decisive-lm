"""SIB-200: topic Choice in many languages, for the general head's non-English calibration."""
import pytest

from decisivelm.data.converters import sib200
from decisivelm.data.dataset import check_licence, load_registry


@pytest.fixture
def rows(tmp_path):
    for lang, texts in {"eng_Latn": ["The match ended in a draw.", "Rain fell on the hills."],
                        "fra_Latn": ["Le match s'est terminé par un nul.", "La pluie est tombée sur les collines."],
                        "pol_Latn": ["Mecz zakończył się remisem.", "Deszcz padał na wzgórza."]}.items():
        d = tmp_path / "data" / lang
        d.mkdir(parents=True)
        (d / "train.tsv").write_text(f"index_id\tcategory\ttext\n1\tsports\t{texts[0]}\n")
        (d / "dev.tsv").write_text(f"index_id\tcategory\ttext\n2\tgeography\t{texts[1]}\n")
        (d / "test.tsv").write_text("index_id\tcategory\ttext\n")
    return {r.lineage.source_id: r for r in sib200.convert(tmp_path)}


def test_suite_languages_are_left_out_and_every_row_is_gold(rows):
    assert set(rows) == {"eng_Latn-1", "eng_Latn-2", "fra_Latn-1", "fra_Latn-2"}
    assert {r.split for r in rows.values()} <= {"gold_calib", "gold_test"}
    assert rows["fra_Latn-1"].language == "fra_Latn" and rows["fra_Latn-1"].tier == "S"


def test_a_sentence_has_one_split_and_one_family_in_every_language(rows):
    assert rows["eng_Latn-1"].split == rows["fra_Latn-1"].split
    assert rows["eng_Latn-1"].lineage.family_id == rows["fra_Latn-1"].lineage.family_id


def test_target_is_the_published_topic(rows):
    p = rows["fra_Latn-1"].targets["topic"].human.probabilities
    assert p["sports"] == 1.0 and sum(p.values()) == 1.0 and len(p) == 7


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
