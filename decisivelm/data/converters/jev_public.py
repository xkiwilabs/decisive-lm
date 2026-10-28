"""Public Jev benchmarks as test-only rows: the exact items and question wording independent benchmarks sent to Jev.

Each benchmark published Jev's aggregate scores on public datasets; re-running the same items, with the same
instructions and option descriptions, through our open models gives a like-for-like comparison without ever calling
Jev or reading its outputs (docs/guide/16 §16.1). Data comes from the original Hub datasets (configs/datasets.yaml `fetch`),
never from the benchmark repos, several of which also store Jev's raw answers.

Every row is `gold_test` with a one-hot human target. configs/holdout.yaml keeps these sources out of the general
adapter, so its comparison with Jev stays zero-shot.

- dhruvmehra/jevbench (MIT): banking77, AG News, SST-2. 500 items per dataset: the evaluated split in Hub order,
  shuffled with `random.Random(0)`, first 500 (its `datasets.load`). One Choice question `label`.
- elcronos/jev-vs-open-decision-models: emotion, tweet_topic, fin_topic, daily_dialog, each whole evaluated split in
  file order, text unmodified. One Choice question whose options are the label names with empty descriptions (its
  headline "plain" variant), plus emotion's "defined" variant with its label definitions as a second file.
"""
import json
import pathlib
import random

import pandas as pd

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

# --- dhruvmehra/jevbench, src/jevbench/datasets.py (MIT), verbatim -------------------------------------------------
DHRUV_INSTRUCTIONS = "Which category best describes the text? Pick exactly one."
DHRUV_N, DHRUV_SEED = 500, 0

BANKING77_RAW = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

DHRUV = {
    # name: (file, text field, {label id: description} in Hub label-index order)
    "bench_sst2": ("validation.parquet", "sentence", {
        "negative": "The sentence expresses negative sentiment about the movie.",
        "positive": "The sentence expresses positive sentiment about the movie.",
    }),
    "bench_agnews": ("test.parquet", "text", {
        "world": "World news: international affairs, politics, governments, elections, wars, conflicts, disasters and crime. Not business, sports or technology.",
        "sports": "Sports news: games, matches, athletes, teams, leagues, scores, transfers and results.",
        "business": "Business news: stock markets, earnings, economy, oil and commodities, trade, mergers, airlines, retail and companies outside the technology sector.",
        "sci_tech": "Science and technology news: technology companies and their products, earnings or lawsuits (e.g. Microsoft, Google, Apple, IBM, Intel), software, hardware, internet, telecom, video games, space, and scientific or medical research.",
    }),
    "bench_banking77": ("test.parquet", "text", {
        r.lower().replace("?", ""): "Customer asks about: " + r.lower().replace("?", "").replace("_", " ")
        for r in BANKING77_RAW
    }),
}
LICENSES = {"bench_sst2": ("LicenseRef-unknown", "X"), "bench_agnews": ("LicenseRef-unknown", "X"),
            "bench_banking77": ("CC-BY-4.0", "R")}


# --- elcronos/jev-vs-open-decision-models, datasets_registry.py and models/common.py, verbatim -------------------------
ELCRONOS = {
    # name: (question id, instructions, labels in dataset id order)
    "bench_emotion": ("emotion", "Which single primary emotion is expressed in this text?",
                      ["sadness", "joy", "love", "anger", "fear", "surprise"]),
    "bench_tweet_topic": ("topic", "Which single topic does this tweet belong to?",
                          ["arts & culture", "business & entrepreneurs", "pop culture", "daily life", "sports & gaming",
                           "science & technology"]),
    "bench_fin_topic": ("topic", "Which single topic does this financial news tweet belong to?",
                        ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
                         "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
                         "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
                         "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]),
    "bench_daily_dialog": ("emotion", "Which single emotion is expressed in this utterance?",
                           ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]),
}
EMOTION_DEFINITIONS = {
    "sadness": "feeling unhappy, down, grief, loss, disappointment or loneliness",
    "joy": "feeling happy, pleased, cheerful, content or excited",
    "love": "feeling affection, warmth, tenderness, caring or romantic attachment",
    "anger": "feeling mad, irritated, annoyed, resentful or hostile",
    "fear": "feeling afraid, scared, anxious, nervous or worried",
    "surprise": "feeling amazed, shocked, startled or caught off guard by something unexpected",
}
N_ROWS = {"bench_emotion": 2000, "bench_tweet_topic": 1693, "bench_fin_topic": 4117, "bench_daily_dialog": 7740}
LICENSES.update({"bench_emotion": ("LicenseRef-other", "X"), "bench_tweet_topic": ("LicenseRef-other", "X"),
                 "bench_fin_topic": ("MIT", "X"), "bench_daily_dialog": ("CC-BY-NC-SA-4.0", "X")})


def elcronos_items(name: str, source_dir: pathlib.Path) -> list[tuple[str, str, int]]:
    """(source id, text, gold label index) in the benchmark's row order."""
    d = pathlib.Path(source_dir)
    if name == "bench_emotion":
        df = pd.read_parquet(d / "test.parquet")
        items = [(f"test-{i:05d}", t, int(y)) for i, (t, y) in enumerate(zip(df.text, df.label))]
    elif name == "bench_tweet_topic":
        rows = [json.loads(line) for line in open(d / "test_2021.single.json", encoding="utf-8")]
        labels = ELCRONOS[name][2]
        for r in rows:
            if r["label_name"].replace("_", " ") != labels[r["label"]]:
                raise ValueError(f"tweet_topic label {r['label']} is {r['label_name']!r}, expected {labels[r['label']]!r}")
        items = [(f"test_2021-{i:05d}", r["text"], int(r["label"])) for i, r in enumerate(rows)]
    elif name == "bench_fin_topic":
        df = pd.read_csv(d / "topic_valid.csv")
        items = [(f"validation-{i:05d}", t, int(y)) for i, (t, y) in enumerate(zip(df.text, df.label))]
    elif name == "bench_daily_dialog":
        df = pd.read_parquet(d / "test.parquet")
        items, k = [], 0
        for dialog, emotions in zip(df.dialog, df.emotion):
            for turn, (utterance, y) in enumerate(zip(dialog, emotions)):
                items.append((f"test-{k:05d}", str(utterance), int(y)))   # verbatim, trailing space kept
                k += 1
    else:
        raise ValueError(name)
    if len(items) != N_ROWS[name]:
        raise ValueError(f"{name}: {len(items)} rows, expected {N_ROWS[name]}")
    return items


def convert_elcronos(name: str, source_dir: pathlib.Path, variant: str = "plain") -> list[Record]:
    qid, instructions, labels = ELCRONOS[name]
    if variant == "plain":
        criteria = {label: "" for label in labels}
    elif variant == "defined" and name == "bench_emotion":
        criteria = {label: EMOTION_DEFINITIONS[label] for label in labels}
    else:
        raise ValueError(f"{name} has no {variant!r} variant")
    license, tier = LICENSES[name]
    items = [(sid, text, labels[y]) for sid, text, y in elcronos_items(name, source_dir)]
    return records(name, license, tier, qid, instructions, criteria, items, f"elcronos-{variant}")


def seeded_sample(n_rows: int, n: int, seed: int) -> list[int]:
    """Row indices in the order jevbench evaluates them: `random.Random(seed).shuffle` of the split, first n.
    Shuffling indices permutes exactly as shuffling the example list does (the swaps depend only on length)."""
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    return order[:n]


def records(source: str, license: str, tier: str, qid: str, instructions: str, criteria: dict[str, str | None],
            items: list[tuple[str, str, str]], benchmark: str) -> list[Record]:
    """Rows from (source id, text, gold label id)."""
    question = {"type": "choice", "instructions": instructions, "criteria": criteria}
    return [Record(
        id=f"{source}-{sid}",
        lineage=Lineage(source_dataset=source, source_id=sid, family_id=f"{source}-{benchmark}"),
        license=license, tier=tier, modality=["text"], language="en", state=str(text),
        questions={qid: question},
        targets={qid: QuestionTargets(human=HumanTarget(probabilities={c: float(c == gold) for c in criteria}))},
        preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
        split="gold_test") for sid, text, gold in items]


def convert_dhruv(name: str, source_dir: pathlib.Path) -> list[Record]:
    file, field, criteria = DHRUV[name]
    df = pd.read_parquet(pathlib.Path(source_dir) / file)
    ids = list(criteria)
    split = file.split(".")[0]
    items = [(f"{split}-{i:05d}", df[field].iloc[i], ids[int(df["label"].iloc[i])])
             for i in seeded_sample(len(df), DHRUV_N, DHRUV_SEED)]
    license, tier = LICENSES[name]
    return records(name, license, tier, "label", DHRUV_INSTRUCTIONS, criteria, items, "dhruvmehra-jevbench")
