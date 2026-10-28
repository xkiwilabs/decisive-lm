# 5. Writing good questions

DecisiveLM answers exactly the questions you write, and nothing else, so the wording of a question decides most of
how useful its answer is. This chapter is practical guidance for writing questions that work well. Most of it follows
from one design rule: **every option, level and statement is judged on its own**, together with the state and its
question's instructions, and never alongside its siblings.

## 5.1 How the model reads a question

For each question the model sees the following, and only this:

| Part | Seen by |
|---|---|
| The state (and any images) | every option, level and statement of every question |
| The question's `instructions` | every option or level of **that** question |
| One option's name and description, one level's description, or one Noul statement with its criteria | only that option, level or statement |

It never sees:

- the **question id**;
- the **other options** of a Choice, or the other levels of a Score;
- an option's **position**, or a level's number;
- **other questions** and their answers.

Chapter [6](06-how-it-works.md) explains the mechanism. In practice it means:

- **The instructions must carry the whole question.** `"department": {"instructions": "Which one?"}` gives the model
  nothing: it never sees the word "department". Write `"Which team should handle this support message?"`.
- **Every description must stand alone.** A description that refers to its neighbours ("as above, but worse", "the
  remaining cases") cannot be understood, because the neighbours are not visible.
- **Rules shared by all options belong in the instructions.** Precedence rules ("if a response mentions both A and B,
  code it as A"), definitions and the coding manual go there once, and every option of that question sees them.

## 5.2 Choosing the type

| Use | When | Example |
|---|---|---|
| **Choice** | Exactly one of several unordered categories applies. | Routing department, intent, document type, language. |
| **Score** | The answer is a position on an ordered scale. | Severity, frustration, quality, agreement (Likert). |
| **Noul** | A single statement is more or less true. Several may be true at once. | "Mentions a deadline", "Contains personal data", "Is spam". |

Some guidance for the borderline cases:

- **Ordered categories are a Score, not a Choice.** "Low / medium / high" as a Choice loses the order. As a Score it
  gives an expected level (`score`) and spreads uncertainty between neighbouring levels, which is what people mean by
  "between medium and high".
- **Several labels that can co-occur are several Nouls.** A Choice's probabilities sum to 1, so it cannot say "both".
  If a text can be both a complaint and a question, ask two Nouls.
- **A yes/no question can be a Noul or a two-option Choice.** Prefer the Noul: it is one block instead of two and
  returns the probability directly. Use a Choice when the two outcomes are genuinely different categories with their
  own descriptions (`{"approve": "...", "escalate": "..."}`).

## 5.3 One Choice or a set of Nouls for a codebook

A coding scheme (a codebook) can be asked in two ways:

1. **One Choice** with one option per code. The answer is a single code with a distribution over all codes. This fits
   when codes are mutually exclusive and exhaustive, as in most content-analysis codebooks with a precedence rule.
2. **One Noul per code** ("The response expresses code A as defined: …"). Each code gets its own probability, and
   several can be high. This fits multi-label coding, and lets you review each code separately.

For a single-label codebook, prefer the Choice:

- put the full codebook and its precedence rules in `instructions`;
- give every option a self-contained definition in its description;
- the softmax then weighs the codes against each other, which a set of independent Nouls cannot do.

## 5.4 Choice options

- **Name and description both reach the model**, as `Option: {name} — {description}`. Use a readable name
  (`technical`, not `opt_2`), and put the definition in the description.
- **Descriptions should be parallel and specific.** Give each option a definition of similar length and detail. A
  much richer description for one option can draw probability towards it for reasons unrelated to the state.
- **Self-contained `other` options.** An `other` option cannot see what "the above" is. Say what it covers in its own
  words:

  | Weak | Better |
  |---|---|
  | `"other": "Other"` | `"other": "The message is about something other than billing, bugs or pricing, such as a job application or a press enquiry"` |
  | `"none": "None of the above"` | `"none": "The response gives no reason for the choice, or only restates the question"` |

  `lint_request` warns about bare `other`-style options (chapter [4](04-requests-and-responses.md), section 4.6).
- **Near-duplicates need a distinguishing clause.** If two options are easily confused, say in each description what
  separates it from the other, without naming the other option by position: `"refund": "Asks for money back for a
  purchase (not a cancellation of a future payment)"`.
- **Large Choices.** Up to 255 options are allowed. Every option is one block, so cost grows with the number of
  options and the length of their descriptions. Keep descriptions short when the list is long. Over 255 options, split
  the decision: a coarse Choice first, then a finer one per group in a second request.

## 5.5 Score levels

- **Describe every level fully.** A level never sees its number or its neighbours, so `"2"`, `"Medium"` or `"More
  than level 1"` carry little information. Write what the state looks like at that level:

  | Weak | Better |
  |---|---|
  | `["Low", "Medium", "High"]` | `["Calm, just stating facts", "Frustrated but civil: complains, stays polite", "Very angry: insults, threats or strong language"]` |

- **Order from lowest to highest.** The array position is the level number, and `score` is the expected level, so the
  direction must be consistent across your questions if you compare scores.
- **Keep levels distinct and roughly evenly spaced.** Overlapping level descriptions spread probability across them,
  which lowers confidence for a reason that has nothing to do with the state.
- **Between 2 and 10 levels.** Most human rating scales use 3 to 7. More levels than people can reliably tell apart
  add cost without adding information.

## 5.6 Noul statements

A Noul's `instructions` are rendered as a **statement** (`Statement: … .`), so write them as a claim about the state,
not as a question:

| Weak | Better |
|---|---|
| `"Urgent?"` | `"The message conveys urgency or time-sensitivity"` |
| `"Does it mention money"` | `"The text states a specific amount of money"` |

Use `criteria` when the boundary needs explaining:

- a **string** adds criteria for the statement as a whole: `"criteria": "Asking about the refund policy without
  requesting a refund does not count"`;
- `{"yes": ..., "no": ...}` says what each outcome means. It is useful when "no" covers several cases:
  `{"yes": "The reviewer says others should buy it", "no": "They advise against it, or give no recommendation"}`.

One statement per Noul. "The message is urgent and from a paying customer" asks two things. Its probability is hard
to read when only one of them holds.

## 5.7 Structured instructions and criteria

Instructions and criteria may be JSON objects or arrays. They reach the model as compact JSON, in the order written.
This is useful for definitions with examples:

```json
"criteria": {
  "complaint": {"what": "The customer reports a problem with a product or service",
                "examples": ["The parcel arrived damaged", "I was charged twice"]},
  "question":  {"what": "The customer asks for information without reporting a problem",
                "examples": ["Do you ship to Canada?"]}
}
```

Keep examples short and clearly labelled. They are read with every state, so long example lists slow every request
down.

## 5.8 States

- **Text or JSON.** Pass free text as a string. Pass a record as a JSON object; its keys are visible to the model, so
  readable key names help (`"customer_message"`, not `"f3"`).
- **Length.** States over 8,192 tokens are cut to their first 70% and last 30% of the budget (chapter
  [4](04-requests-and-responses.md), section 4.3). If the important part of a long document sits in the middle,
  extract it before sending.
- **Keep the state about one thing.** Ask about one message, one document or one response per request. The questions
  then have one unambiguous subject.
- **Do not put earlier model answers in the state** unless the question is about them. They become evidence the model
  reads, and can make answers self-reinforcing.

## 5.9 Many questions in one request

Questions are independent. A question's answer is the same whether it is asked alone or with fifty others, so you can
group questions freely:

- **Ask everything about a state in one request.** The state is processed once and shared by all questions, so each
  extra question costs only its own instructions and options.
- **Adding a question never changes the others.** You can add a new question to a production request without
  re-validating the existing ones.
- **One answer is never context for another.** If question B depends on the answer to question A, ask A first and put
  its answer in B's state or instructions in a second request.

## 5.10 A checklist

Before using a question at scale:

1. Would a person who sees only the instructions and one option know what is being asked?
2. Does each description, level and statement make sense with its siblings hidden?
3. Is there a self-contained `other` option where the list may not be exhaustive?
4. Are ordered answers a Score, and co-occurring labels separate Nouls?
5. Has it been tried on a few dozen real states, with the least confident answers read by hand? Chapter
   [11](11-evaluation.md) shows how to measure whether confidence points at the right items. If it does not,
   chapter [9](09-fine-tuning-for-your-task.md) shows how to fine-tune on your own labels.

Next: [6. How it works](06-how-it-works.md)
