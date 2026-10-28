import collections
import datetime

import pytest

from decisivelm.api_types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from decisivelm.synth import generate as g
from decisivelm.synth.teacher import Gate, Schedule


def test_spec_is_seeded_and_follows_the_type_mix():
    assert g.sample_spec("a") == g.sample_spec("a")
    specs = [g.sample_spec(f"s{i}") for i in range(3000)]
    types = collections.Counter(q.type for s in specs for q in s.questions if s.setting != g.HIRING)
    n = sum(types.values())
    for t, w in g.TYPE_WEIGHTS.items():
        assert abs(types[t] / n - w) < 0.03
    for s in specs:
        for q in s.questions:
            if q.type == "choice":
                assert 2 <= q.k <= g.MAX_CHOICE
            if q.type == "score":
                assert 2 <= q.k <= 10


def test_hiring_specs_are_factual_nouls_only():
    hiring = [s for s in (g.sample_spec(f"h{i}") for i in range(3000)) if s.setting == g.HIRING]
    assert hiring
    assert all(q.type == "noul" for s in hiring for q in s.questions)
    user = g.questions_prompt(hiring[0])[1]["content"]
    assert "Never ask about or infer age, gender" in user


def test_prompt_lists_every_question_in_order():
    s = g.Spec("x", "logistics", "warehouse operations", "incident reports",
               [g.QSpec("choice", 4, "codebook", "other"), g.QSpec("score", 5, "terse"), g.QSpec("noul", None, "terse", "plain")])
    user = g.questions_prompt(s)[1]["content"]
    assert "choice, score, noul" in user
    assert "Question 1: choice with exactly 4 options." in user and "catch-all" in user
    assert "Question 2: score with exactly 5 levels." in user
    assert "Question 3: yes/no statement." in user


def opt(name, desc, examples=None):
    return {"name": name, "description": desc, **({"examples": examples} if examples else {})}


def test_choice_becomes_a_question_with_snake_case_names():
    raw = {"type": "choice", "instructions": "What is the request about?",
           "options": [opt("Billing Issue", "Charges and invoices."), opt("delivery", "Late or lost parcels.")]}
    q, why = g.to_question(raw, g.QSpec("choice", 2))
    assert why is None and isinstance(q, ChoiceQuestion)
    assert q.criteria == {"billing_issue": "Charges and invoices.", "delivery": "Late or lost parcels."}


def test_structured_style_keeps_examples():
    raw = {"type": "score", "instructions": "How urgent is it?",
           "options": [opt("low", "Can wait.", ["next week is fine"]), opt("high", "Needs action today.", ["outage"])]}
    q, _ = g.to_question(raw, g.QSpec("score", 2, "structured"))
    assert isinstance(q, ScoreQuestion)
    assert q.criteria[0] == {"what": "Can wait.", "examples": ["next week is fine"]}


@pytest.mark.parametrize("raw,spec,reason", [
    ({"type": "noul", "instructions": "x", "options": []}, g.QSpec("choice", 2), "type"),
    ({"type": "choice", "instructions": "Pick one.", "options": [opt("a", "A."), opt("b", "B.")]},
     g.QSpec("choice", 3), "options, asked for 3"),
    ({"type": "choice", "instructions": "Pick one.", "options": [opt("a", "A."), opt("A", "B.")]},
     g.QSpec("choice", 2), "duplicate"),
    ({"type": "choice", "instructions": "Pick one.",
      "options": [opt("a", "About cards."), opt("other", "Anything else; none of the above.")]},
     g.QSpec("choice", 2), "refers"),
    ({"type": "choice", "instructions": "Pick one.",
      "options": [opt("a", "About cards."), opt("other", "Use when none of the previous options are suitable.")]},
     g.QSpec("choice", 2), "refers"),
    ({"type": "noul", "instructions": "As in the previous question, is it urgent?", "options": []},
     g.QSpec("noul"), "refers"),
    ({"type": "noul", "instructions": "Is it urgent?", "options": [opt("yes", "Urgent.")]}, g.QSpec("noul"), "noul with"),
])
def test_bad_questions_are_rejected(raw, spec, reason):
    q, why = g.to_question(raw, spec)
    assert q is None and reason in why


def test_noul_with_and_without_criteria():
    plain, _ = g.to_question({"type": "noul", "instructions": "The parcel is damaged.", "options": []}, g.QSpec("noul"))
    assert isinstance(plain, NoulQuestion) and plain.criteria is None
    crit, _ = g.to_question({"type": "noul", "instructions": "The parcel is damaged.",
                             "options": [opt("yes", "Visible damage."), opt("no", "Intact.")]}, g.QSpec("noul"))
    assert crit.criteria.yes == "Visible damage." and crit.criteria.no == "Intact."
    empty, _ = g.to_question({"type": "noul", "instructions": "The parcel is damaged.",
                              "options": [opt("yes", ""), opt("no", " ")]}, g.QSpec("noul"))
    assert isinstance(empty, NoulQuestion) and empty.criteria is None


def test_to_questions_keys_and_counts():
    s = g.Spec("x", "d", "s", "k", [g.QSpec("noul"), g.QSpec("noul")])
    qs, why = g.to_questions({"questions": [{"type": "noul", "instructions": "A.", "options": []},
                                            {"type": "choice", "instructions": "B.", "options": []}]}, s)
    assert list(qs) == ["q0"] and len(why) == 1


def test_states_lose_copied_labels_and_json_parses():
    assert g.to_state("State 3: My parcel never came.", "easy") == "My parcel never came."
    assert g.to_state("**State 1:** Hello", "easy") == "Hello"
    assert g.to_state('{"order": 12, "status": "late"}', "json") == {"order": 12, "status": "late"}
    assert g.to_state('{"order": 12', "json") == '{"order": 12'
    assert g.to_state('{"order": 12}', "easy") == '{"order": 12}'


def test_gate_follows_the_schedule_and_yields_in_the_day():
    s = Schedule(day_hours=(6, 23), day_max_in_flight=4, night_max_in_flight=64)
    assert s.limit(datetime.datetime(2026, 1, 1, 12)) == 4
    assert s.limit(datetime.datetime(2026, 1, 1, 23, 30)) == 64
    assert s.limit(datetime.datetime(2026, 1, 1, 5, 59)) == 64
    gate = Gate(s, waiting=lambda: 3, poll=0)
    assert gate.limit() == (0 if s.is_day() else 64)


def test_version_2_asks_for_codebook_style_catch_alls_and_accepts_them():
    specs = [g.sample_spec(f"v{i}", version=2) for i in range(2000)]
    extras = collections.Counter(q.extra for s in specs for q in s.questions if q.type == "choice")
    assert extras["relative_other"] > 0 and all(s.version == 2 for s in specs)
    assert "relative_other" not in {q.extra for s in (g.sample_spec(f"v{i}") for i in range(2000)) for q in s.questions}
    s = g.Spec("x", "d", "s", "k", [g.QSpec("choice", 3, "terse", "relative_other")], version=2)
    assert "pointing" in g.questions_prompt(s)[1]["content"]
    raw = {"type": "choice", "instructions": "Which department?",
           "options": [opt("billing", "Charges."), opt("delivery", "Parcels."), opt("other", "None of the above.")]}
    q, why = g.to_question(raw, g.QSpec("choice", 3, "terse", "relative_other"))
    assert why is None and q.criteria["other"] == "None of the above."
    raw["options"][0]["description"] = "Charges not covered by the options above."
    assert g.to_question(raw, g.QSpec("choice", 3, "terse", "relative_other"))[0] is None


def test_version_2_draws_a_quarter_from_coding_codebooks_and_version_1_is_unchanged():
    v2 = [g.sample_spec(f"c{i}", version=2) for i in range(2000)]
    coding = [s for s in v2 if g.is_coding(s)]
    assert abs(len(coding) / len(v2) - g.CODING_SHARE) < 0.04
    assert all(5 <= q.k <= 12 for s in coding for q in s.questions if q.type == "choice")
    assert all("video" not in s.setting for s in coding)
    assert not any(g.is_coding(g.sample_spec(f"c{i}")) for i in range(2000))
    assert set(g.sample_profiles("x", 50, coding[0])) <= set(g.CODING_PROFILES)
    assert "terse" in g.templates(2)["profiles"] and "terse" not in g.templates(1)["profiles"]


def test_version_2_plain_nouls_are_not_asked_for_option_descriptions():
    s = g.Spec("x", "d", "s", "k", [g.QSpec("noul", None, "structured", "plain")], version=2)
    line = [l for l in g.questions_prompt(s)[1]["content"].splitlines() if l.startswith("Question 1")][0]
    assert line == "Question 1: yes/no statement. Leave the options empty."


def test_version_3_codes_half_the_sets_with_split_states_and_leaves_version_2_alone():
    v3 = [g.sample_spec(f"t{i}", version=3) for i in range(2000)]
    coding = [s for s in v3 if g.is_coding(s)]
    assert abs(len(coding) / len(v3) - 0.5) < 0.04
    profiles = collections.Counter(p for s in coding for p in g.sample_profiles(s.seed, 6, s))
    assert profiles["split"] > 0 and set(profiles) <= set(g.CODING_PROFILES_V3)
    assert "split" in g.templates(3)["profiles"] and "split" not in g.templates(2)["profiles"]
    v2 = [g.sample_spec(f"t{i}", version=2) for i in range(200)]
    assert all("split" not in g.sample_profiles(s.seed, 6, s) for s in v2)
    rest = [s for s in v3 if not g.is_coding(s)]
    assert "split" not in {p for s in rest for p in g.sample_profiles(s.seed, 6, s)}


def test_version_4_asks_for_plain_scores_with_open_states_and_leaves_version_3_alone():
    v4 = [g.sample_spec(f"u{i}", version=4) for i in range(2000)]
    assert not any(g.is_coding(s) for s in v4)
    scores = [q for s in v4 for q in s.questions if q.type == "score"]
    qs = [q for s in v4 if s.setting != g.HIRING for q in s.questions]
    assert abs(len(scores) / len(qs) - 0.45) < 0.03
    assert {q.extra for q in scores} == {"evidence", "plain_middle", None}
    assert all(q.k in (3, 4, 5) for q in scores if q.extra)
    assert "open" in {p for s in v4[:200] for p in g.sample_profiles(s.seed, 6, s)}
    t = g.templates(4)
    assert "open" in t["profiles"] and "open" not in g.templates(3)["profiles"]
    s = g.Spec("x", "d", "s", "k", [g.QSpec("score", 3, "terse", "evidence")], version=4)
    assert "leaves it open" in g.questions_prompt(s)[1]["content"]
    v3 = [g.sample_spec(f"u{i}", version=3) for i in range(300)]
    assert all(q.extra is None for s in v3 for q in s.questions if q.type == "score")
    assert "open" not in {p for s in v3 for p in g.sample_profiles(s.seed, 6, s)}
