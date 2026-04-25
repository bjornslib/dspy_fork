# Optimizing RLM Policy on LongCoT with GEPA

This guide walks through running a focused 50-question pilot on the
[LongCoT dataset](https://huggingface.co/datasets/LongHorizonReasoning/longcot)
and using GEPA to improve on the baseline published in
[LongCoT — A Benchmark Worthy of an RLM's Attention](https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/).

**Baseline to beat:** RLM 45.4% (227/500), Vanilla 2.6% (13/500), using
`claude-sonnet-4-5`, DSPy 3.1.3, `max_iterations=50`.

---

## Why a 50-question pilot?

A full 500-question run costs ~$621 at baseline rates. GEPA's
`optimize_anything` may call the evaluator 100+ times. A 50-question
subset costs roughly $60 per evaluator call — tolerable for a search
budget of 10–20 calls; prohibitive for 100.

The 25 easy / 25 hard split is deliberate:

- **25 easy** anchor the policy. If a well-tuned policy can't hold 80–90%
  on easy questions it's regressing, not improving.
- **25 hard** provide the gradient signal. The baseline scrores ~10–20% on
  hard questions across most domains; there is real room to move.

---

## Prerequisites

```bash
# 1. DSPy fork with policy-field support (this repo)
pip install -e .

# 2. LongCoT evaluation harness (verifier + scoring)
pip install git+https://github.com/LongHorizonReasoning/longcot.git

# 3. Dataset loader
pip install datasets

# 4. GEPA (optimize_anything)
pip install gepa

# 5. Set your provider key
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: enable the Gemini fallback judge for math/chemistry
export GEMINI_API_KEY="..."
```

---

## Step 1 — Load the dataset and build the 50-question subset

```python
import random
from datasets import load_dataset

SEED = 42
random.seed(SEED)

# Load all domains; splits are "easy", "medium", "hard"
ds = load_dataset("LongHorizonReasoning/longcot", "all")

DOMAINS = ["logic", "cs", "chemistry", "chess", "math"]

def sample_split(split_name: str, n_per_domain: int) -> list[dict]:
    """Return n_per_domain rows per domain from the given split."""
    split = ds[split_name]
    rows = []
    for domain in DOMAINS:
        pool = [r for r in split if r["domain"] == domain]
        rows.extend(random.sample(pool, min(n_per_domain, len(pool))))
    return rows

easy_questions = sample_split("easy", n_per_domain=5)   # 25 total
hard_questions = sample_split("hard", n_per_domain=5)   # 25 total
pilot = easy_questions + hard_questions

print(f"Pilot size: {len(pilot)} questions")
print(f"Domain distribution: { {d: sum(1 for r in pilot if r['domain']==d) for d in DOMAINS} }")
print(f"Difficulty distribution: { {s: sum(1 for r in pilot if r['difficulty']==s) for s in ['easy','hard']} }")
```

### Why stratify by domain?

The blog post shows RLM performance varies wildly by domain:

| Domain     | Baseline RLM | Notes |
|------------|-------------|-------|
| Logic      | ~100%       | Externalises cleanly to code |
| Chess      | 85%         | Good room to improve |
| Chemistry  | 31%         | Strong improvement target |
| CS / DistMem | 16%      | Some signal; MaxFlow/HM at 0% |
| Math       | 6%          | Hardest; policy helps framing |

Sampling 5 per domain per difficulty keeps all five domains in the pilot,
so the optimised policy is general rather than tuned to one domain. If you
only care about one domain, replace `DOMAINS` with a single entry and
increase `n_per_domain` to 25.

> **Tip — reproducibility:** Save the sampled IDs so every GEPA trial uses
> exactly the same 50 questions.
>
> ```python
> import json
> pilot_ids = [r["question_id"] for r in pilot]
> with open("pilot_ids.json", "w") as f:
>     json.dump(pilot_ids, f, indent=2)
> ```
>
> To reload the exact subset later:
>
> ```python
> with open("pilot_ids.json") as f:
>     pilot_ids = json.load(f)
> id_set = set(pilot_ids)
> all_rows = list(ds["easy"]) + list(ds["hard"])
> pilot = [r for r in all_rows if r["question_id"] in id_set]
> ```

---

## Step 2 — Define the signature and module

The blog post uses `prompt -> response` with RLM extracting a
`solution = ...` line. We extend it with the new `policy` field so GEPA
has a handle to optimise.

```python
import dspy

class LongCoTSolve(dspy.Signature):
    """Solve a LongCoT problem.

    The `prompt` already contains the full problem statement and the answer
    format requirement (always ends with `solution = ...`). Reason through
    the problem with the available REPL, then return the final response —
    which MUST contain the literal `solution = ...` line as instructed.
    """
    prompt: str = dspy.InputField(
        desc="Full LongCoT problem prompt with answer-format instructions"
    )
    policy: str = dspy.InputField(
        desc="Operational policy for the REPL agent."
    )
    response: str = dspy.OutputField(
        desc="Full final response containing the required `solution = ...` line"
    )


MAIN_LM = dspy.LM("anthropic/claude-sonnet-4-5", max_tokens=16000)
SUB_LM  = dspy.LM("anthropic/claude-sonnet-4-5", max_tokens=8000)

dspy.configure(lm=MAIN_LM)


class PolicyRLM(dspy.Module):
    def __init__(self, policy: str):
        super().__init__()
        self.policy = policy
        self.agent = dspy.RLM(
            LongCoTSolve,
            max_iterations=20,   # blog used 50; 20 keeps cost down for search
            max_llm_calls=40,
            sub_lm=SUB_LM,
        )

    def forward(self, question: dict) -> dspy.Prediction:
        return self.agent(
            prompt=question["prompt"],
            policy=self.policy,
        )
```

> **On `max_iterations`:** The blog ran `max_iterations=50`. For the
> optimisation loop, 20 is a practical ceiling — it keeps each evaluator
> call under ~3 minutes while still giving the agent enough steps to solve
> most easy questions. Raise it back to 50 for your final evaluation run.

---

## Step 3 — Seed policy

Start from the policy developed in this repo. The seed describes general
REPL discipline; GEPA will refine the domain-specific heuristics.

```python
SEED_POLICY = """
You are operating a Python REPL to solve a difficult long-horizon reasoning task.

Rules:
1. Read the problem completely before writing any code. Identify the domain
   (logic, chess, chemistry, math, CS) and choose the right strategy.
2. Prefer exact, deterministic Python computation when the answer can be
   derived symbolically or by simulation.
3. Use llm_query only for semantic judgments that cannot be encoded in code
   — e.g. checking chemical SMILES validity or interpreting ambiguous text.
4. Batch related llm_query calls with llm_query_batched to stay within the
   sub-LLM budget.
5. Print intermediate results at every step. Never assume a computation
   is correct without seeing its output.
6. The answer MUST be assigned as `solution = ...`. Verify the format
   matches what the prompt specifies before submitting.
7. If an approach produces a wrong or empty result, backtrack and try an
   alternative rather than submitting a guess.
8. Submit only when the solution is grounded in observed REPL output.
""".strip()
```

---

## Step 4 — Evaluation metric

The `longcot` package ships domain-specific verifiers. Each verifier
parses the `solution = ...` line from the model's response and checks it
against the canonical answer.

```python
import longcot

def score_response(question: dict, response: str) -> float:
    """Return 1.0 if correct, 0.0 otherwise. Never raises."""
    try:
        return 1.0 if longcot.verify(question, response) else 0.0
    except Exception:
        return 0.0


def summarize_trace(pred: dspy.Prediction) -> dict:
    """Extract trace signals for GEPA's reflection step."""
    traj = getattr(pred, "trajectory", []) or []
    last = traj[-1] if traj else {}
    response = getattr(pred, "response", "")
    has_solution_line = "solution =" in response
    return {
        "num_steps":        len(traj),
        "has_solution_line": has_solution_line,
        "last_reasoning":   str(last.get("reasoning", ""))[:800],
        "last_code":        str(last.get("code", ""))[:1000],
        "last_output":      str(last.get("output", ""))[:1000],
        "response_tail":    response[-500:],
    }
```

> **Fallback judge:** For math and chemistry, `longcot.verify` activates a
> Gemini fallback when `GEMINI_API_KEY` is set and the primary verifier is
> inconclusive. Set it for the most accurate scores; omit it for cheaper
> runs where approximate scores are good enough for GEPA's search.

---

## Step 5 — Baseline run (reproduce the blog post on your pilot)

Run the baseline without GEPA first to establish your starting score on
the 50 questions.

```python
import json

def run_baseline(policy: str, questions: list[dict]) -> dict:
    program = PolicyRLM(policy=policy)
    results = []
    for q in questions:
        pred    = program.forward(q)
        response = getattr(pred, "response", "")
        score   = score_response(q, response)
        results.append({
            "question_id": q["question_id"],
            "domain":      q["domain"],
            "difficulty":  q["difficulty"],
            "score":       score,
            "response":    response,
            **summarize_trace(pred),
        })
        print(f"  {q['question_id'][:30]:<30} {'✓' if score else '✗'}  "
              f"steps={results[-1]['num_steps']}  sol={'yes' if results[-1]['has_solution_line'] else 'NO'}")

    correct = sum(r["score"] for r in results)
    print(f"\nBaseline: {correct:.0f}/{len(results)} = {correct/len(results):.1%}")
    by_domain = {d: [r for r in results if r["domain"]==d] for d in DOMAINS}
    for d, rs in by_domain.items():
        c = sum(r["score"] for r in rs)
        print(f"  {d:<12} {c:.0f}/{len(rs)}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


baseline_results = run_baseline(SEED_POLICY, pilot)
```

Expected ballpark on this stratified sample given the blog post numbers:

| Domain    | Easy (5 q) | Hard (5 q) |
|-----------|-----------|-----------|
| Logic     | ~5/5      | ~4/5      |
| Chess     | ~4/5      | ~3/5      |
| Chemistry | ~2/5      | ~1/5      |
| CS        | ~1/5      | ~0/5      |
| Math      | ~1/5      | ~0/5      |
| **Total** | **~13/25**| **~8/25** |

A combined pilot baseline of roughly **21/50 (42%)** would be consistent
with the published 45.4% — close enough given sample variance.

---

## Step 6 — GEPA optimisation loop

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig


def evaluate_policy(candidate_policy: str) -> float:
    """Evaluator passed to GEPA. Returns solved rate on the 50-question pilot."""
    program = PolicyRLM(policy=candidate_policy)
    scores  = []

    for q in pilot:
        pred     = program.forward(q)
        response = getattr(pred, "response", "")
        score    = score_response(q, response)
        scores.append(score)

        trace = summarize_trace(pred)
        oa.log({
            "question_id":    q["question_id"],
            "domain":         q["domain"],
            "difficulty":     q["difficulty"],
            "score":          score,
            **trace,
        })

    solved_rate = sum(scores) / len(scores)
    print(f"  Candidate score: {solved_rate:.1%}  ({sum(scores):.0f}/{len(scores)})")
    return solved_rate


result = optimize_anything(
    seed_candidate=SEED_POLICY,
    evaluator=evaluate_policy,
    objective=(
        "Improve the solved rate on a mixed LongCoT benchmark covering logic, "
        "chess, chemistry, computer science, and mathematics. The agent uses a "
        "Python REPL with access to llm_query. Focus on: efficient domain "
        "detection at the start, correct solution-line formatting, avoiding "
        "wasted llm_query calls on computable sub-problems, and backtracking "
        "when an approach stalls rather than submitting a guess."
    ),
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=30)  # ~$60 × 30 ≈ $1 800 total
    ),
)

print("\n=== Best policy ===")
print(result.best_candidate)
print(f"\nBest pilot score: {result.best_score:.1%}")

with open("best_policy.txt", "w") as f:
    f.write(result.best_candidate)
```

### GEPA budget guide

| `max_metric_calls` | Approx cost (50 q, claude-sonnet-4-5) | Use case |
|-------------------|--------------------------------------|----------|
| 10                | ~$600                                | Quick proof of concept |
| 30                | ~$1 800                              | Solid search (recommended) |
| 100               | ~$6 000                              | Full optimisation |

Costs assume ~$0.12/question at `max_iterations=20`. Adjust your
`max_iterations` and `max_llm_calls` to control spend per question.

---

## Step 7 — Validate on the held-out set

Never report the GEPA-optimised score from the training pilot — it will be
inflated. Evaluate the best policy on a separate held-out sample drawn
from the `medium` split (which the optimiser never saw).

```python
holdout = sample_split("medium", n_per_domain=5)  # 25 questions

holdout_results = run_baseline(result.best_candidate, holdout)
pilot_results   = run_baseline(result.best_candidate, pilot)

print(f"\nPilot (in-distribution):  {sum(r['score'] for r in pilot_results)/len(pilot_results):.1%}")
print(f"Holdout (medium, unseen): {sum(r['score'] for r in holdout_results)/len(holdout_results):.1%}")
```

---

## Step 8 — Full 500-question run (optional)

Once you have a validated best policy, run the full benchmark to get a
comparable number to the blog post's 227/500.

```python
all_questions = list(ds["easy"]) + list(ds["medium"]) + list(ds["hard"])
# Or just the 500-question "mini" sample from the blog post — filter to
# the templates used there if you want an exact apples-to-apples comparison.

full_results = run_baseline(result.best_candidate, all_questions)
```

---

## Interpreting results and next steps

### What GEPA is likely to improve

Based on the failure modes in the blog post:

| Task | Primary failure | Policy lever |
|------|----------------|--------------|
| Chess | Misformatting the move sequence | Explicit format-verification rule |
| Chemistry | Incorrect SMILES construction | Prefer `rdkit` over manual string-building |
| Math | Giving up after one failed approach | Backtracking rule with alternative strategy |
| CS / DistMem | Wrong algorithm choice | Domain-detection heuristic at step 1 |

Tasks where policy optimisation is unlikely to help: **MaxFlow-MinCut** and
**Hindley-Milner** (0/75 baseline). These are likely architectural — the
agent lacks the right search primitives, not the right behavioural norms.

### If scores don't improve

- Check `has_solution_line` in the trace logs. If `False`, the agent is
  computing a correct answer but failing to format it — add an explicit
  formatting rule to the seed policy before re-running GEPA.
- Check `num_steps`. If most runs hit `max_iterations`, the agent is
  running out of time, not ideas — raise `max_iterations` or add an
  early-commitment heuristic to the policy.
- If the pilot score improves but the holdout score doesn't, GEPA has
  overfit to the 50 questions — increase pilot size to 100 or constrain
  the objective to domain-general language.

### Splitting into `action_policy` + `extract_policy`

Once the single `policy` field converges, a natural next step is to split
it into two separately optimisable fields that map to RLM's two internal
stages:

```python
class LongCoTSolveV2(dspy.Signature):
    """Solve a LongCoT problem."""
    prompt: str         = dspy.InputField(desc="Full LongCoT problem prompt")
    action_policy: str  = dspy.InputField(desc="Policy for iterative REPL reasoning")
    extract_policy: str = dspy.InputField(desc="Policy for formatting the final solution line")
    response: str       = dspy.OutputField(desc="Full response containing `solution = ...`")
```

Run GEPA on `action_policy` first (holds `extract_policy` fixed), then
alternate. This reduces the search space per stage and tends to converge
faster than jointly optimising a single large policy string.

---

## Quick reference

```bash
# Run everything end-to-end
python examples/longcot_policy_optimization.py

# Evaluate a saved policy against the full benchmark
python -c "
import json, longcot, dspy
from examples.longcot_policy_optimization import PolicyRLM, pilot, score_response

with open('best_policy.txt') as f:
    policy = f.read()

results = [score_response(q, PolicyRLM(policy).forward(q).response) for q in pilot]
print(f'Score: {sum(results)}/{len(results)} = {sum(results)/len(results):.1%}')
"
```

---

## Sources

- [LongCoT dataset — Hugging Face](https://huggingface.co/datasets/LongHorizonReasoning/longcot)
- [LongCoT — A Benchmark Worthy of an RLM's Attention](https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/)
- [LongCoT paper (arXiv 2604.14140)](https://huggingface.co/papers/2604.14140)
- [LongCoT evaluation harness](https://github.com/LongHorizonReasoning/longcot)
