#!/usr/bin/env python3

import re
import json
import argparse

import rich
from rich.console import Console


console = Console()


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    if isinstance(data, dict):
        lst = []
        for k, v in data.items():
            v["id"] = k
            lst.append(v)
        return lst


def compute_format_score(text: str) -> float:
    """
    Compute a reward score for how well the given text matches the expected format:
    <think>...
    <parallel>
      <goal>
        <outline>...</outline>
        <outline>...</outline>
        <outline>...</outline>
      </goal>
      <path>...</path>
      <path>...</path>
      <path>...</path>
    </parallel>
    ...
    <parallel>
      <goal>...</goal>
      <path>...</path>
    </parallel>
    ...
    </think>
    ...
    \\boxed{}

    Requirements:
    - All required tags must appear in the correct nested structure.
    - Multiple <parallel>...</parallel> blocks allowed inside <think>.
    - Each <parallel> must contain exactly one <goal>...</goal> and its matching <path>...</path> tags.
    - <outline> and <path> counts must match within each block.
    - Tags must appear in correct order.
    - Content-free tag pairs must be adjacent.
    - Final text must include exactly one \\boxed{...} with non-empty content.

    Returns a float reward.
    """

    score = 0.0

    # Normalize whitespace
    text = re.sub(r"\s+", "", text.strip())

    # Match top-level <think>...</think>
    think_pattern = r"^<think>(.*?)</think>(.*)$"
    think_match = re.match(think_pattern, text, re.DOTALL)
    if not think_match:
        # print("Format error: Missing or invalid <think>...</think> structure.")
        return -2.0

    def has_invalid_content(section: str) -> bool:
        special_tags = [
            "<goal>",
            "</goal>",
            "<outline>",
            "</outline>",
            "<path>",
            "</path>",
            "<parallel>",
            "</parallel>",
            "<think>",
            "</think>",
        ]
        for tag in special_tags:
            if tag in section:
                return True
        return False

    inside_think, after_think = [g.strip() for g in think_match.groups()]

    if after_think and has_invalid_content(after_think):
        # Content after </think> is allowed but must not contain special tags
        # print("Format error: Invalid content after </think>.")
        return -0.5

    # Parse multiple <parallel>...</parallel> blocks
    parallel_pattern = r"<parallel>(.*?)</parallel>"
    parallel_blocks = list(re.finditer(parallel_pattern, inside_think, re.DOTALL))
    if not parallel_blocks:
        # print("Format error: No <parallel>...</parallel> blocks found.")
        return -1.5

    pos = 0
    for pb in parallel_blocks:
        if pb.start() != pos:
            extra = inside_think[pos : pb.start()].strip()
            if has_invalid_content(extra):
                # print("Format error: Invalid content between <parallel> blocks.")
                return -0.5
        pos = pb.end()

        block_content = pb.group(1)

        # Match one <goal>...</goal> followed by <path>...</path>+
        block_pattern = r"^<goal>(.*?)</goal>((?:<path>.*?</path>)+)$"
        block_match = re.match(block_pattern, block_content, re.DOTALL)
        if not block_match:
            # print("Format error: Invalid structure inside <parallel>.")
            return -1.0

        goal_content, following = block_match.groups()

        # Validate outlines
        outlines = re.findall(r"<outline>(.*?)</outline>", goal_content, re.DOTALL)
        if len(outlines) == 0:
            # print("Format error: No <outline> tags found inside <goal>.")
            return -1.0

        # Ensure only outlines inside goal
        outlines_concat = "".join(f"<outline>{o}</outline>" for o in outlines)
        if outlines_concat.strip() != goal_content.strip():
            # print("Format error: Non-outline content inside <goal>.")
            score -= 0.4

        # Validate adjacency for outlines
        outline_rejoined = re.sub(
            r"<outline>.*?</outline>",
            "</outline><outline>",
            goal_content,
            flags=re.DOTALL,
        )
        cleaned_outline = outline_rejoined.replace("</outline><outline>", "").strip()
        if cleaned_outline:
            # print("Format error: <outline> tags are not adjacent.")
            score -= 0.2

        # Extract paths immediately following this goal
        path_iter = list(re.finditer(r"<path>.*?</path>", following, re.DOTALL))
        paths = [m.group(0) for m in path_iter]
        path_content = "".join(paths)

        if path_content.strip() != following.strip():
            # print("Format error: Non-path content after </goal>.")
            score -= 0.4

        # Parse path contents
        path_texts = re.findall(r"<path>(.*?)</path>", following, re.DOTALL)
        if len(path_texts) != len(outlines):
            # print("Format error: Number of <path> does not match number of <outline>.")
            return -1.0

        # Validate adjacency for paths
        path_rejoined = re.sub(
            r"<path>.*?</path>", "</path><path>", following, flags=re.DOTALL
        )
        cleaned_path = path_rejoined.replace("</path><path>", "").strip()
        if cleaned_path:
            # print("Format error: <path> tags are not adjacent.")
            score -= 0.2

        # Too short checks
        for i, outline in enumerate(outlines):
            if len(outline) < 3:
                # print(f"Format error: Outline {i} too short.")
                score -= 0.4
        for i, path in enumerate(path_texts):
            if len(path) < 6:
                # print(f"Format error: Path {i} too short.")
                score -= 0.4

    # Verify \\boxed{}
    pattern = re.compile(r"\\boxed\{(.*?)\}")
    matches = pattern.findall(text)
    count = len(matches)
    has_empty_content = any(m.strip() == "" for m in matches) if count > 0 else False
    if has_empty_content:  # count != 1 only for law
        # print("Format error: \\boxed{} must be non-empty content.")
        score -= 1.0

    return score


def compute_multiverse_format_score(s):
    think_start = r"<Think>"
    think_end = r"</Think>"
    parallel_start = r"<Parallel>"
    parallel_end = r"</Parallel>"
    goal_start = r"<Goal>"
    goal_end = r"</Goal>"
    outline_start = r"<Outline>"
    outline_end = r"</Outline>"
    path_start = r"<Path>"
    path_end = r"</Path>"
    conclusion_start = r"<Conclusion>"
    conclusion_end = r"</Conclusion>"

    # Ensure exactly one \boxed{} with non-empty content
    pattern = re.compile(r"\\boxed\{([^{}]*)\}")
    matches = pattern.findall(s)
    count = len(matches)
    has_empty_content = any(m.strip() == "" for m in matches) if count > 0 else False
    if count == 0 or has_empty_content:
        # breakpoint()
        return 0

    # Ensure all tags are present and correctly appeared without ordered
    for tag in [
        think_start,
        think_end,
        parallel_start,
        parallel_end,
        goal_start,
        goal_end,
        outline_start,
        outline_end,
        path_start,
        path_end,
        conclusion_start,
        conclusion_end,
    ]:
        if tag not in s:
            # breakpoint()
            return 0

    return 1


def compute_multiverse_format_scores(data):
    good_format_and_correct = 0
    bad_format_but_correct = 0
    good_format_but_incorrect = 0
    bad_format_and_incorrect = 0
    # good_format_and_correct: well formatted and correct
    # bad_format_but_correct: badly formatted but correct
    # good_format_but_incorrect: well formatted but incorrect
    # bad_format_and_incorrect: badly formatted and incorrect
    for dp in data:
        gold = dp["reference"].strip()
        answers = [d.strip() for d in dp["boxed_preds"]]
        preds = dp["full_preds"]
        samples_size = len(answers)
        # format_score = compute_multiverse_format_score(pred)
        cleaned_preds = [
            "<think>" + pred.split("<think>", 1)[-1].strip() for pred in preds
        ]
        format_scores = []
        for cleaned_pred in cleaned_preds:
            format_scores.append(compute_format_score(cleaned_pred))

        target_format_score = 0.0
        for answer, format_score in zip(answers, format_scores):
            if answer == gold:
                acc = 1.0
            else:
                acc = 0.0

            if format_score == target_format_score and acc == 1.0:
                good_format_and_correct += 1
            elif format_score < target_format_score and acc == 1.0:
                bad_format_but_correct += 1
            elif format_score == target_format_score and acc == 0.0:
                good_format_but_incorrect += 1
            elif format_score < target_format_score and acc == 0.0:
                bad_format_and_incorrect += 1
            else:
                raise ValueError(
                    f"Invalid format score or accuracy: {format_score}, {acc}"
                )

    assert (
        good_format_and_correct
        + bad_format_but_correct
        + good_format_but_incorrect
        + bad_format_and_incorrect
        == len(data) * samples_size
    )

    console.print(f"Format scores: {format_scores}")

    well_formatted_pred_nums = sum(format_scores)
    console.print(
        f"Well formatted vs. Bad cases (% ratio): {well_formatted_pred_nums} / {len(data)} ({well_formatted_pred_nums / len(data):.2%}))"
    )
    # create a table to show the above statistics
    table = rich.table.Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", width=40)
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_row(
        "Well formatted and correct",
        str(good_format_and_correct),
        f"{good_format_and_correct / (len(data) * samples_size):.2%}",
    )
    table.add_row(
        "Bad format but correct",
        str(bad_format_but_correct),
        f"{bad_format_but_correct / (len(data) * samples_size):.2%}",
    )
    table.add_row(
        "Well formatted but incorrect",
        str(good_format_but_incorrect),
        f"{good_format_but_incorrect / (len(data) * samples_size):.2%}",
    )
    table.add_row(
        "Bad format and incorrect",
        str(bad_format_and_incorrect),
        f"{bad_format_and_incorrect / (len(data) * samples_size):.2%}",
    )
    console.print(table)
    return format_scores


def main(args: argparse.Namespace):
    data = load_json(args.input_file)
    scores = compute_multiverse_format_scores(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="prediction/Multiverse-32B/zero_shot/AIME25_1.json",
        help="Path to the input JSON file containing model predictions.",
    )
    args = parser.parse_args()
    main(args)
