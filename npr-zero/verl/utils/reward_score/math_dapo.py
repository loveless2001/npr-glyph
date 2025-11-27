# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional

from verl.utils.reward_score.grader import math_equal
from verl.utils.reward_score.parser import extract_answer


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def get_format_reward(text: str) -> float:
    """
    <guideline>
    <plan>
    1:...
    </plan>
    <plan>
    2:...
    </plan>
    <plan>
    3:...
    </plan>
    </guideline>
    <step>
    1:...
    </step>
    <step>
    2:...
    </step>
    <step>
    3:...
    </step>
    <takeaway>
    ...
    </takeaway>
    <guideline>
    <plan>
    1:...
    </plan>
    </guideline>
    <step>
    1:...
    </step>
    <takeaway>
    ...
    </takeaway>
    \boxed{concise final answer}

    Rules:
    * All tags must appear exactly in the order shown.
    * Each '<guideline>' contains only '<plan>' tags.
    * The number of '<step>' tags must exactly match the number of '<plan>' tags.
    * The contents between each <step> are independent of each other.
    * '<plan>' and '<step>' must be adjacent with no extra content.
    * Content inside '<plan>' and '<step>' must be meaningful (not empty).
    * The contents between each <step> are independent of each other and cannot know each other's contents.
    * Include exactly one non-empty '\boxed{...}' containing the short final answer.

    Returns a float reward.
    """

    score = 0.0

    # Normalize whitespace
    text = re.sub(r"\s+", "", text.strip())

    def has_invalid_content(section: str) -> bool:
        special_tags = ["<guideline>", "</guideline>", "<plan>", "</plan>", "<step>", "</step>", "<takeaway>", "</takeaway>"]
        for tag in special_tags:
            if tag in section:
                return True
        return False

    # Parse multiple guideline-step-takeaway blocks
    # Each block starts with <guideline>...</guideline>, followed by <step>...</step>+, and ends with <takeaway>...</takeaway>
    block_pattern = r"<guideline>(.*?)</guideline>((?:(?!<guideline>).)*?)<takeaway>(.*?)</takeaway>"
    block_matches = list(re.finditer(block_pattern, text, re.DOTALL))
    if not block_matches:
        return -1.5

    pos = 0
    for bm in block_matches:
        if bm.start() != pos:
            extra = text[pos: bm.start()].strip()
            if has_invalid_content(extra):
                return -1.5
        pos = bm.end()

        guideline_content, steps_block, takeaway_content = bm.groups()

        # Validate plans inside guideline
        plans = re.findall(r"<plan>(.*?)</plan>", guideline_content, re.DOTALL)
        if len(plans) == 0:
            return -1.0

        # Ensure only plans inside guideline
        plans_concat = "".join(f"<plan>{p}</plan>" for p in plans)
        if plans_concat.strip() != guideline_content.strip():
            score -= 0.4

        # Validate adjacency for plans
        plan_rejoined = re.sub(r"<plan>.*?</plan>", "</plan><plan>", guideline_content, flags=re.DOTALL)
        cleaned_plan = plan_rejoined.replace("</plan><plan>", "").strip()
        if cleaned_plan:
            score -= 0.2

        # Extract step contents
        step_texts = re.findall(r"<step>(.*?)</step>", steps_block, re.DOTALL)
        if len(step_texts) != len(plans):
            return -1.0

        # Validate adjacency for steps
        step_rejoined = re.sub(r"<step>.*?</step>", "</step><step>", steps_block, flags=re.DOTALL)
        cleaned_step = step_rejoined.replace("</step><step>", "").strip()
        if cleaned_step:
            score -= 0.2

        # Too short checks
        for plan in plans:
            if len(plan) < 3 or has_invalid_content(plan):
                score -= 0.4
        for step in step_texts:
            if len(step) < 6 or has_invalid_content(plan):
                score -= 0.4

        # Takeaway length check
        if len(takeaway_content.strip()) < 6 or has_invalid_content(plan):
            score -= 0.4

    if pos < len(text):
        extra = text[pos:].strip()
        if has_invalid_content(extra):
            return -1.0

        # Verify \\boxed{}
        pattern = re.compile(r'\\boxed\{(.*?)\}')
        matches = pattern.findall(extra)
        count = len(matches)
        has_empty_content = any(m.strip() == "" for m in matches) if count > 0 else True
        if has_empty_content:
            return -1.0

    return score


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """

    if solution_str == "":
        return {
            "score": -2.0,
            "acc": 0.0,
            # "format_reward": -2.0,
            # "No_format_correct": 0.0
        }

    # format_reward = get_format_reward(solution_str)

    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    if not correct:
        pred = extract_answer(solution_str, data_name="AIME25", use_last_number=True)
        correct = math_equal(pred, ground_truth)

    reward = 1.0 if correct else -1.0
    acc = correct

    # No_format_correct = 0.0
    # if format_reward < 0 and correct:
    #     No_format_correct = 1.0
    
    # if format_reward != 0.0:
    #     reward = 0.0
    #     acc = 0.0

    return {
        "score": reward,  #  + format_reward,
        "acc": float(acc),
        # "format_reward": format_reward,
        # "No_format_correct": No_format_correct
    }
