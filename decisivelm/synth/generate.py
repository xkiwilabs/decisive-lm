"""Teacher-written questions and states (docs/guide/13 §13.4; also fan-out on real states).

A seeded `Spec` fixes everything that should vary: domain, setting, state kind, how many questions, their types
(~40% Noul, ~35% Choice, ~25% Score), Choice sizes (long tail, mostly 3–15) and Score levels, and each question's
description style (terse, descriptive, codebook, structured examples), self-contained `other` options and
near-duplicates. The teacher only writes text to that spec, as JSON under a schema, so variety comes from the spec.

Spec versions: 1 (prompts/generate_v1.yaml) asks for self-contained catch-alls only; 2 (generate_v2.yaml) also asks
for catch-alls written the way people write them ("None of the above"); 3 adds coding sets with split answers; 4
(generate_v4.yaml) asks for plainly worded Scores with a middle level, and states that bear on them without settling
them. A leaf never sees its siblings, so the
student has to learn a low prior for such an option; the teacher sees every option and labels them as written.

Generated Choices have at most MAX_CHOICE options, part of every spec version's distribution (changing it would change
the specs a seed produces). Larger Choices come from the seeds and schema explosion; the labeller can score them.
Hiring questions are factual Nouls about documents only (docs/guide/14 §14.1, docs/guide/16 §16.5).
"""
import dataclasses
import hashlib
import json
import random
import re

import yaml

from decisivelm.api_types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from decisivelm.synth.label import PROMPTS

MAX_CHOICE = 20
TYPE_WEIGHTS = {"noul": 0.40, "choice": 0.35, "score": 0.25}
CHOICE_SIZES = [2] * 6 + [3] * 12 + [4] * 14 + [5] * 12 + [6] * 10 + [7] * 8 + [8] * 7 + [9] * 5 + [10] * 5 + \
    [11, 12, 12, 13, 14, 15, 15, 16, 18, 20]
SCORE_LEVELS = [2] * 3 + [3] * 18 + [4] * 12 + [5] * 30 + [6] * 5 + [7] * 12 + [10] * 8 + [8, 9]
STYLES = {"terse": 0.25, "descriptive": 0.35, "codebook": 0.25, "structured": 0.15}
PROFILES = ("easy", "borderline", "ambiguous", "irrelevant", "long", "json")
PROFILE_WEIGHTS = (0.25, 0.25, 0.2, 0.1, 0.1, 0.1)

# (domain, settings, state kinds). Settings make the same domain ask different things.
DOMAINS = [
    ("customer support", ["telecom provider help desk", "airline customer care", "software-as-a-service support",
                          "utility company billing queries", "online marketplace seller support"],
     ["customer emails", "live chat messages", "support tickets with subject and body"]),
    ("banking and finance", ["retail bank fraud team", "credit card disputes", "small-business lending desk",
                             "personal finance app", "insurance claims intake"],
     ["customer messages", "transaction descriptions", "claim notes written by agents"]),
    ("content moderation", ["online forum", "product review site", "children's education platform",
                            "news comment section", "dating app profiles"],
     ["user posts", "comments", "profile descriptions"]),
    ("documents and forms", ["accounts payable", "rental applications", "government permit office",
                             "university admissions office", "procurement"],
     ["invoices as text", "filled-in application forms", "scanned letters transcribed to text"]),
    ("security and guardrails", ["chat assistant safety filter", "security operations centre",
                                 "email phishing triage", "code review for secrets"],
     ["user prompts sent to an assistant", "alert descriptions from monitoring tools", "suspicious emails",
      "code snippets"]),
    ("retail and e-commerce", ["fashion retailer", "grocery delivery", "electronics store returns",
                               "furniture marketplace"],
     ["product reviews", "return requests", "product listings"]),
    ("logistics", ["parcel delivery company", "freight forwarding", "warehouse operations", "fleet maintenance"],
     ["delivery exception notes", "shipment emails", "driver incident reports"]),
    ("healthcare administration", ["clinic appointment line", "hospital patient feedback", "pharmacy orders",
                                   "health insurance pre-authorisation"],
     ["patient messages to reception", "feedback forms", "administrative notes"]),
    ("legal", ["contract review", "employment tribunal intake", "tenancy disputes", "privacy compliance"],
     ["contract clauses", "client intake summaries", "privacy policy excerpts"]),
    ("education", ["essay marking for secondary school", "university course feedback", "online course forum",
                   "language learning app"],
     ["student answers to short-answer questions", "course evaluations", "forum questions"]),
    ("qualitative research coding", ["survey on remote work", "interviews about public transport",
                                     "open responses about a health campaign", "user research on a mobile app",
                                     "free-text answers in a psychology experiment"],
     ["open-ended survey responses", "interview excerpts", "participants' free-text descriptions"]),
    ("human resources", ["employee engagement survey", "internal IT helpdesk", "workplace incident reporting",
                         "hiring paperwork checks"],
     ["employee comments", "helpdesk requests", "incident reports", "CVs and cover letters"]),
    ("real estate", ["property listings site", "building maintenance requests", "strata management"],
     ["listing descriptions", "maintenance requests", "owner correspondence"]),
    ("travel and hospitality", ["hotel guest reviews", "travel agency bookings", "restaurant reservations"],
     ["guest reviews", "booking change requests", "customer messages"]),
    ("software engineering", ["bug tracker triage", "app store reviews", "pull request review",
                              "incident post-mortems"],
     ["bug reports", "app reviews", "commit messages and diffs", "log excerpts"]),
    ("media and publishing", ["newsroom wire triage", "podcast listener mail", "academic journal submissions"],
     ["news headlines with lead paragraphs", "listener emails", "paper abstracts"]),
    ("public services", ["city council complaints line", "emergency non-urgent line", "public library enquiries"],
     ["resident complaints", "call summaries", "enquiry emails"]),
    ("energy and environment", ["solar installer leads", "environmental incident reports", "smart-meter support"],
     ["customer enquiries", "incident reports", "sensor event logs"]),
    ("agriculture and food", ["farm supply orders", "food safety inspections", "restaurant health complaints"],
     ["orders", "inspection notes", "complaints"]),
    ("sports and recreation", ["gym membership support", "sports club registrations", "fan forum"],
     ["emails from members", "registration forms", "fan posts"]),
]
HIRING = "hiring paperwork checks"

# Spec version 2 draws a quarter of its question sets from qualitative coding of short free-text answers, written as
# codebooks (decision rules, inclusion/exclusion, examples; 5–12 codes, often a "None of the above"), with states
# as participants type them. Descriptions of what a person in a video intends are left out: that is the coding
# task's scenario (results/general_v1.md), which stays a zero-shot test.
CODING_SHARE = 0.25
CODING = ("qualitative coding of open-ended answers",
          ["survey on remote work", "interviews about public transport", "open responses about a health campaign",
           "user research on a mobile app", "free-text answers in a psychology experiment",
           "customer exit survey asking why people cancelled", "teachers' reflections on a classroom lesson",
           "open answers about climate worries", "patients describing how a condition affects daily life",
           "participants explaining the choices they made in a decision-making study",
           "students' answers to 'what did you learn this week?'", "museum visitors' comment cards",
           "volunteers describing why they volunteer", "app users' answers to 'what nearly stopped you signing up?'",
           "residents' answers to a council survey about their neighbourhood",
           "participants' descriptions of their morning routine in a time-use study"],
          ["one-line answers typed by study participants", "open-ended survey responses",
           "short interview excerpts", "participants' free-text descriptions"])
CODING_TYPES = {"choice": 0.6, "noul": 0.25, "score": 0.15}
CODING_STYLES = {"codebook": 0.6, "descriptive": 0.25, "structured": 0.15}
CODING_EXTRAS = {None: 0.35, "relative_other": 0.35, "other": 0.15, "near_duplicate": 0.15}
CODING_PROFILES = {"terse": 0.35, "easy": 0.15, "borderline": 0.2, "ambiguous": 0.2, "irrelevant": 0.1}
# v3: more of the answers coders disagree on, including "split" answers that fit two codes about equally well (what
# ρ with human agreement and tie AUROC measure; General v2's ρ on the coding task was 0.335, the frozen E4B's 0.385).
CODING_PROFILES_V3 = {"terse": 0.25, "easy": 0.1, "borderline": 0.2, "ambiguous": 0.2, "split": 0.2,
                      "irrelevant": 0.05}
PROFILE_WEIGHTS_V3 = (0.15, 0.3, 0.3, 0.1, 0.1, 0.05)
# v4 (2026-10-24): General v3c's Scores on a new task leaned to the top level. On MultiNLI's 3-level Score, where the
# raters' answer was "neither shows the hypothesis true nor false", it chose that level 31% of the time and leaned to
# "It shows the hypothesis is true" (P0: 73%); with the levels worded "definitely true/false given the premise" it
# chose the middle 67% of the time (results/dev_review_general_v3c_e4b_r8_seed0_nt.md and the 2026-10-24 change log).
# A level is read without its siblings, so "shows it is true" has to be judged strictly on its own. v4 asks for
# more Scores, many of them scales of how far the state shows something, worded plainly, with a middle level, and adds
# an "open" state profile: related to a Score's claim and consistent with its higher levels, but not settling it.
# No coding sets: General v3c is the last mix chosen with the coding task.
PROFILES_V4 = PROFILES + ("open",)
PROFILE_WEIGHTS_V4 = (0.2, 0.2, 0.15, 0.05, 0.1, 0.05, 0.25)
TYPE_WEIGHTS_V4 = {"noul": 0.3, "choice": 0.25, "score": 0.45}
SCORE_EXTRAS_V4 = {"evidence": 0.5, "plain_middle": 0.3, None: 0.2}
EVIDENCE_LEVELS = [3] * 3 + [4] + [5] * 2


@dataclasses.dataclass
class QSpec:
    type: str
    k: int | None = None          # options (Choice) or levels (Score)
    style: str = "descriptive"
    extra: str | None = None      # Choice: other / near_duplicate; Noul: criteria / plain; Score (v4): evidence / plain_middle


@dataclasses.dataclass
class Spec:
    seed: str
    domain: str
    setting: str
    state_kind: str
    questions: list[QSpec]
    version: int = 1


# Per spec version: the prompt file, the Choice extras with their weights (3+ options), the share of question sets
# from the coding domain, and the state profile weights (coding sets, the rest).
VERSIONS = {
    1: {"prompt": "generate_v1", "choice_extra": {None: 0.6, "other": 0.25, "near_duplicate": 0.15},
        "coding_share": 0.0, "profile_weights": PROFILE_WEIGHTS},
    2: {"prompt": "generate_v2",
        "choice_extra": {None: 0.5, "other": 0.15, "relative_other": 0.2, "near_duplicate": 0.15},
        "coding_share": CODING_SHARE, "coding_profiles": CODING_PROFILES, "profile_weights": PROFILE_WEIGHTS},
    3: {"prompt": "generate_v3",
        "choice_extra": {None: 0.5, "other": 0.15, "relative_other": 0.2, "near_duplicate": 0.15},
        "coding_share": 0.5, "coding_profiles": CODING_PROFILES_V3, "profile_weights": PROFILE_WEIGHTS_V3},
    4: {"prompt": "generate_v4",
        "choice_extra": {None: 0.5, "other": 0.15, "relative_other": 0.2, "near_duplicate": 0.15},
        "coding_share": 0.0, "profiles": PROFILES_V4, "profile_weights": PROFILE_WEIGHTS_V4,
        "type_weights": TYPE_WEIGHTS_V4, "score_extra": SCORE_EXTRAS_V4},
}


def rng_for(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def _weighted(rng, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def sample_qspec(rng, qtype: str, version: int = 1) -> QSpec:
    style = _weighted(rng, STYLES)
    if qtype == "choice":
        k = rng.choice(CHOICE_SIZES)
        extra = _weighted(rng, VERSIONS[version]["choice_extra"]) if k >= 3 else None
        return QSpec("choice", k, style, extra)
    if qtype == "score":
        if "score_extra" not in VERSIONS[version]:
            return QSpec("score", rng.choice(SCORE_LEVELS), style)
        extra = _weighted(rng, VERSIONS[version]["score_extra"])
        return QSpec("score", rng.choice(EVIDENCE_LEVELS if extra else SCORE_LEVELS), style, extra)
    return QSpec("noul", None, style, rng.choices(["plain", "criteria"], weights=[0.6, 0.4])[0])


def sample_coding_qspec(rng, qtype: str) -> QSpec:
    style = _weighted(rng, CODING_STYLES)
    if qtype == "choice":
        return QSpec("choice", rng.randint(5, 12), style, _weighted(rng, CODING_EXTRAS))
    if qtype == "score":
        return QSpec("score", rng.choice(SCORE_LEVELS), style)
    return QSpec("noul", None, style, rng.choices(["plain", "criteria"], weights=[0.5, 0.5])[0])


def is_coding(spec: "Spec") -> bool:
    return spec.domain == CODING[0]


def sample_spec(seed: str, n_range=(3, 8), version: int = 1) -> Spec:
    rng = rng_for(seed)
    share = VERSIONS[version]["coding_share"]
    if share and rng.random() < share:
        domain, settings, kinds = CODING
        setting, kind, n = rng.choice(settings), rng.choice(kinds), rng.randint(2, 5)
        qs = [sample_coding_qspec(rng, _weighted(rng, CODING_TYPES)) for _ in range(n)]
        return Spec(seed, domain, setting, kind, qs, version)
    domain, settings, kinds = rng.choice(DOMAINS)
    setting = rng.choice(settings)
    kind = rng.choice(kinds)
    if setting == HIRING:
        kind = "CVs and cover letters"
    n = rng.randint(*n_range)
    weights = VERSIONS[version].get("type_weights", TYPE_WEIGHTS)
    types = ["noul"] * n if setting == HIRING else [_weighted(rng, weights) for _ in range(n)]
    return Spec(seed, domain, setting, kind, [sample_qspec(rng, t, version) for t in types], version)


def templates(version: int = 1) -> dict:
    return yaml.safe_load((PROMPTS / f"{VERSIONS[version]['prompt']}.yaml").read_text())


def _details(spec: Spec, t: dict) -> str:
    lines = []
    for i, q in enumerate(spec.questions, 1):
        what = {"noul": "yes/no statement", "choice": f"choice with exactly {q.k} options",
                "score": f"score with exactly {q.k} levels"}[q.type]
        extra = (t["choice_extra"].get(q.extra, "") if q.type == "choice" else
                 t["noul_extra"].get(q.extra, "") if q.type == "noul" else
                 t.get("score_extra", {}).get(q.extra, "") if q.extra else "")
        style = t["style"][q.style]
        if spec.version >= 2 and q.type == "noul" and q.extra == "plain":
            style = ""   # no options to describe (v1 sent both lines)
        lines.append(" ".join(x for x in (f"Question {i}: {what}.", style, extra) if x))
    if spec.setting == HIRING:
        lines.append("Only ask factual questions about what the document says (for example whether it lists a "
                     "skill or a qualification). Never ask about or infer age, gender, ethnicity, health or any "
                     "other personal attribute, and never ask whether to hire.")
    return "\n".join(lines)


def _types(spec: Spec) -> str:
    return ", ".join(q.type for q in spec.questions)


def questions_prompt(spec: Spec) -> list[dict]:
    t = templates(spec.version)
    user = t["questions"].format(domain=spec.domain, setting=spec.setting, state_kind=spec.state_kind,
                                 n=len(spec.questions), types=_types(spec), details=_details(spec, t))
    return [{"role": "system", "content": t["system"]}, {"role": "user", "content": user}]


def fanout_prompt(spec: Spec, state: str, source: str) -> list[dict]:
    t = templates(spec.version)
    user = t["fanout"].format(source=source, state=state, n=len(spec.questions), types=_types(spec),
                              details=_details(spec, t))
    return [{"role": "system", "content": t["system"]}, {"role": "user", "content": user}]


def sample_profiles(seed: str, n: int, spec: "Spec | None" = None) -> list[str]:
    rng = rng_for(seed + ":profiles")
    v = VERSIONS[spec.version if spec is not None else 1]
    if spec is not None and is_coding(spec):
        return rng.choices(list(v["coding_profiles"]), weights=list(v["coding_profiles"].values()), k=n)
    return rng.choices(v.get("profiles", PROFILES), weights=v["profile_weights"], k=n)


def states_prompt(spec: Spec, questions: dict, profiles: list[str]) -> list[dict]:
    t = templates(spec.version)
    shown = "\n".join(f"- {json.dumps(q.model_dump(exclude_none=True), ensure_ascii=False)}" for q in questions.values())
    listed = "\n".join(f"State {i}: {t['profiles'][p]}" for i, p in enumerate(profiles, 1))
    user = t["states"].format(domain=spec.domain, setting=spec.setting, questions=shown, n=len(profiles),
                              state_kind=spec.state_kind, profiles=listed)
    return [{"role": "system", "content": t["system"]}, {"role": "user", "content": user}]


OPTION = {"type": "object", "additionalProperties": False, "required": ["name", "description"],
          "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                         "examples": {"type": "array", "items": {"type": "string"}}}}
QUESTIONS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["type", "instructions", "options"],
        "properties": {"type": {"enum": ["noul", "choice", "score"]}, "instructions": {"type": "string"},
                       "options": {"type": "array", "items": OPTION}}}}}}
STATES_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["states"],
    "properties": {"states": {"type": "array", "items": {"type": "string"}}}}

_SNAKE = re.compile(r"[^a-z0-9]+")
# A leaf never sees its siblings or other questions (docs/guide/06 §6.8), so text that points at them cannot be answered.
_CROSS_REF = re.compile(r"\b(previous|next|above|below|following|other) questions?\b|\bquestion \d|\boption [a-z]\b"
                        r"|none of the above|\bthe above\b|\b(other|previous|preceding|remaining) (options|categories|choices)\b"
                        r"|\b(options|categories|choices) (above|listed)\b|\blisted (options|categories)\b",
                        re.IGNORECASE)


def snake(name: str) -> str:
    return _SNAKE.sub("_", name.strip().lower()).strip("_")[:48]


def _content(o: dict, structured: bool):
    desc = o["description"].strip()
    ex = [e.strip() for e in o.get("examples") or [] if e.strip()]
    return {"what": desc, "examples": ex} if structured and ex else desc


def to_question(raw: dict, q: QSpec):
    """A validated api_types question from the teacher's JSON, or a reason it is rejected."""
    if raw["type"] != q.type:
        return None, f"type {raw['type']} != {q.type}"
    instructions = raw["instructions"].strip()
    if not instructions:
        return None, "empty instructions"
    texts = [instructions] + [o["description"] for o in raw["options"]]
    if q.extra == "relative_other":
        texts = texts[:-1]   # the catch-all was asked to point at the other options; nothing else may
    if any(_CROSS_REF.search(t) for t in texts):
        return None, "refers to other questions or options"
    opts, structured = raw["options"], q.style == "structured"
    if q.type == "noul":
        if len(opts) == 0 or not any(o["description"].strip() for o in opts):   # yes/no named, nothing said
            return NoulQuestion(type="noul", instructions=instructions), None
        if len(opts) != 2 or not all(o["description"].strip() for o in opts):
            return None, f"noul with {len(opts)} options"
        return NoulQuestion(type="noul", instructions=instructions,
                            criteria={"yes": opts[0]["description"].strip(), "no": opts[1]["description"].strip()}), None
    if len(opts) != q.k:
        return None, f"{len(opts)} options, asked for {q.k}"
    if any(not o["description"].strip() for o in opts):
        return None, "empty description"
    if q.type == "score":
        return ScoreQuestion(type="score", instructions=instructions,
                             criteria=[_content(o, structured) for o in opts]), None
    names = [snake(o["name"]) for o in opts]
    if any(not n for n in names) or len(set(names)) != len(names):
        return None, "missing or duplicate option names"
    return ChoiceQuestion(type="choice", instructions=instructions,
                          criteria={n: _content(o, structured) for n, o in zip(names, opts)}), None


def to_questions(raw: dict, spec: Spec) -> tuple[dict, list[str]]:
    """Questions keyed q0, q1, … (ids are never rendered) and the reasons for any dropped."""
    out, why = {}, []
    got = raw.get("questions", [])
    if len(got) != len(spec.questions):
        why.append(f"{len(got)} questions, asked for {len(spec.questions)}")
    for i, (r, q) in enumerate(zip(got, spec.questions)):
        question, reason = to_question(r, q)
        if question is None:
            why.append(reason)
        else:
            out[f"q{i}"] = question
    return out, why


_STATE_LABEL = re.compile(r"^\s*\**\s*state\s*\d+\s*\**\s*[:.\-–]\s*\**\s*", re.IGNORECASE)


def to_state(text: str, profile: str):
    """JSON-profile states become dicts when they parse as an object; everything else stays text. A leading
    "State 3:" that the teacher sometimes copies from the prompt is removed."""
    text = _STATE_LABEL.sub("", text.strip()).strip()
    if profile == "json":
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            pass
    return text
