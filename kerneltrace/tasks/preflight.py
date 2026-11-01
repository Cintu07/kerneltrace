"""task validity gates.

there are two. the structural gate only needs the tokenizer: single-token labels,
equal lengths, exactly one differing position. the behavioural gate needs the
clean and corrupt margins, because the model has to actually do the task on a
case before an intervention on it means anything. rejected cases keep their
reason and stay in the reported denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ioi import IOICase


@dataclass
class TokenizedCase:
    case: IOICase
    clean_ids: list[int]
    corrupt_ids: list[int]
    answer_id: int
    distractor_id: int

    @property
    def length(self) -> int:
        return len(self.clean_ids)


def single_token(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def structural_check(case: IOICase, tokenizer, max_seq_len: int) -> tuple[TokenizedCase | None, list[str]]:
    reasons = []
    a_id = single_token(tokenizer, case.answer)
    b_id = single_token(tokenizer, case.distractor)
    if a_id is None:
        reasons.append("answer_not_single_token")
    if b_id is None:
        reasons.append("distractor_not_single_token")
    if a_id is not None and a_id == b_id:
        reasons.append("answer_equals_distractor")
    clean = tokenizer.encode(case.clean, add_special_tokens=False)
    corrupt = tokenizer.encode(case.corrupt, add_special_tokens=False)
    if len(clean) != len(corrupt):
        reasons.append("length_mismatch")
    else:
        diff = sum(x != y for x, y in zip(clean, corrupt))
        if diff != 1:
            reasons.append(f"differs_at_{diff}_positions_expected_1")
    if len(clean) > max_seq_len:
        reasons.append("exceeds_max_seq_len")
    if reasons:
        return None, reasons
    return TokenizedCase(case, clean, corrupt, a_id, b_id), []


def behavioural_check(clean_margin: float, corrupt_margin: float, min_contrast: float,
                      require_corrupt_flip: bool) -> list[str]:
    reasons = []
    if not clean_margin > 0:
        reasons.append("clean_does_not_prefer_answer")
    if require_corrupt_flip and not corrupt_margin < 0:
        reasons.append("corrupt_does_not_prefer_distractor")
    if clean_margin - corrupt_margin < min_contrast:
        reasons.append("contrast_below_threshold")
    return reasons
