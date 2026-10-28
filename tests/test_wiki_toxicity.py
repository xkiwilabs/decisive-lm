"""Wikipedia Talk Labels, Toxicity: a 5-level Score with many raters, a development check that is never trained on."""
import pytest

from decisivelm.data.converters import wiki_toxicity
from decisivelm.data.dataset import check_licence, load_registry


@pytest.fixture
def rows(tmp_path):
    ann = ["rev_id\tworker_id\ttoxicity\ttoxicity_score"]
    ann += [f"101.0\t{w}\t{int(s < 0)}\t{s}.0" for w, s in enumerate([-2, -1, -1, 0])]   # 4 raters
    ann += [f"102.0\t{w}\t0\t1.0" for w in range(3)]                                     # 3 raters
    ann += [f"103.0\t{w}\t0\t0.0" for w in range(2)]                                     # too few raters
    ann += [f"104.0\t{w}\t0\t2.0" for w in range(5)]                                     # a train comment
    (tmp_path / "toxicity_annotations.tsv").write_text("\n".join(ann) + "\n")
    com = ["rev_id\tcomment\tyear\tlogged_in\tns\tsample\tsplit",
           "101.0\tYou are `wrong`.NEWLINE_TOKENNEWLINE_TOKENStop.\t2015\t1\tuser\tblocked\ttest",
           "102.0\tThanks for the fix.\t2014\t1\tarticle\trandom\ttest",
           "103.0\tOk.\t2014\t1\tarticle\trandom\ttest",
           "104.0\tFine by me.\t2013\t0\tarticle\trandom\ttrain"]
    (tmp_path / "toxicity_annotated_comments.tsv").write_text("\n".join(com) + "\n")
    return {r.lineage.source_id: r for r in wiki_toxicity.convert(tmp_path)}


def test_only_test_split_comments_with_three_raters_are_kept_and_all_are_gold(rows):
    assert set(rows) == {"101", "102"}
    assert {r.split for r in rows.values()} <= {"gold_calib", "gold_test"}


def test_target_is_the_raters_shares_ordered_from_very_toxic(rows):
    t = rows["101"].targets["toxicity"].human
    assert t.probabilities == {"0": 0.25, "1": 0.5, "2": 0.25, "3": 0.0, "4": 0.0} and t.n_raters == 4
    assert rows["102"].targets["toxicity"].human.probabilities["3"] == 1.0
    assert rows["101"].questions["toxicity"].criteria[0].startswith("Very toxic")


def test_the_release_tokens_are_put_back(rows):
    assert rows["101"].state == 'You are "wrong".\n\nStop.'


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
