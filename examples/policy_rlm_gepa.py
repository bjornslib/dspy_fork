"""
Example: Optimizing RLM agent behavior with a GEPA-tunable policy field.

dspy.RLM supports a special ``policy`` input field.  When your signature
includes a field named ``policy``, RLM places it as the *first* input in
every ``generate_action`` call — before ``variables_info`` and history — so
the LLM treats it as a first-class behavioral directive rather than a regular
data variable.

This makes the policy text a natural optimization target for GEPA's
``optimize_anything``, which can refine free-form text artifacts via
evaluate-and-reflect cycles.

Sections
--------
1. Wiring up the policy field
2. PolicyRLM module wrapper
3. GEPA optimization loop
"""

# =============================================================================
# Section 1 – Wiring up the policy field
# =============================================================================
import dspy


class LongCoTTask(dspy.Signature):
    question: str = dspy.InputField()
    policy: str = dspy.InputField(desc="Operational policy for the REPL agent.")
    answer: str = dspy.OutputField()


# RLM detects the 'policy' field and gives it a dedicated action-prompt slot.
rlm = dspy.RLM(LongCoTTask, max_iterations=20, max_llm_calls=40)

# Confirm field ordering: 'policy' is first (before variables_info).
print(list(rlm.generate_action.signature.input_fields.keys()))
# Expected: ['policy', 'variables_info', 'repl_history', 'iteration']

# 'policy' is also absent from the REPL data-variable listing in instructions:
assert "`policy`" not in rlm.generate_action.signature.instructions
assert "`question`" in rlm.generate_action.signature.instructions


# =============================================================================
# Section 2 – PolicyRLM module wrapper
# =============================================================================

SEED_POLICY = """
You are operating a Python REPL to solve a difficult reasoning task.

Rules:
1. Inspect the problem structure before committing to a plan.
2. Prefer deterministic Python computation when the answer can be derived exactly.
3. Use llm_query only for semantic judgments that cannot be encoded directly.
4. Batch related sub-queries whenever multiple similar judgments are needed.
5. Keep intermediate variables explicit and verify assumptions before submitting.
6. Submit only when the final answer is grounded and concise.
7. If uncertain, spend another step checking instead of guessing.
""".strip()


class PolicyRLM(dspy.Module):
    """Wraps RLM with a configurable policy string suitable for GEPA optimization."""

    def __init__(self, lm, sub_lm=None, max_iterations=20, max_llm_calls=40):
        super().__init__()
        self.agent = dspy.RLM(
            LongCoTTask,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm,
        )
        self.agent.set_lm(lm)

    def forward(self, question: str, policy: str) -> dspy.Prediction:
        return self.agent(question=question, policy=policy)


# =============================================================================
# Section 3 – GEPA optimization loop
# =============================================================================
# Requires: pip install gepa
# Set OPENAI_API_KEY (or the relevant provider key) before running.

# Uncomment and fill in `trainset` to run live:
#
# import gepa.optimize_anything as oa
# from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig
#
# MAIN_LM = dspy.LM("openai/gpt-4.1")
# SUB_LM  = dspy.LM("openai/gpt-4.1-mini")
#
# trainset = [...]   # list of dspy.Example(question=..., answer=...)
#
#
# def exact_match(example, pred) -> float:
#     gold  = str(example.answer).strip()
#     guess = str(getattr(pred, "answer", "")).strip()
#     return 1.0 if guess == gold else 0.0
#
#
# def summarize_trace(pred) -> dict:
#     """Extract trace signals for GEPA's reflection step."""
#     traj = getattr(pred, "trajectory", []) or []
#     last = traj[-1] if traj else {}
#     return {
#         "num_steps":      len(traj),
#         "last_reasoning": str(last.get("reasoning", ""))[:1000],
#         "last_code":      str(last.get("code",      ""))[:1200],
#         "last_output":    str(last.get("output",    ""))[:1200],
#         "final_answer":   str(getattr(pred, "answer", ""))[:500],
#     }
#
#
# def evaluate_policy(candidate_policy: str) -> float:
#     """Run the full RLM trace for each example and return the solved rate."""
#     program = PolicyRLM(MAIN_LM, sub_lm=SUB_LM)
#     scores = []
#     for ex in trainset:
#         pred  = program(question=ex.question, policy=candidate_policy)
#         score = exact_match(ex, pred)
#         scores.append(score)
#         # Log rich trace signals so GEPA can reflect on failure modes.
#         oa.log({
#             "question": ex.question[:300],
#             "gold":     str(ex.answer)[:200],
#             "score":    score,
#             **summarize_trace(pred),
#         })
#     return sum(scores) / max(len(scores), 1)
#
#
# result = optimize_anything(
#     seed_candidate=SEED_POLICY,
#     evaluator=evaluate_policy,
#     objective=(
#         "Improve LongCoT solved rate by refining the RLM operational policy. "
#         "Encourage efficient REPL search, selective llm_query usage, fewer wasted "
#         "steps, and more reliable exact final answers."
#     ),
#     config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
# )
#
# print("Best policy:\n", result.best_candidate)
# print("Best score:", result.best_score)
#
#
# Next step: split 'policy' into 'action_policy' + 'extract_policy' to give
# GEPA separate handles for iterative REPL behavior vs. final answer extraction.
