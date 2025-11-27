#!/usr/bin/env python

import os
import re
import json
import uuid
import asyncio
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Optional

import torch
import sglang
import sglang.srt.entrypoints.engine
from rich.syntax import Syntax
from rich.console import Console
from transformers import AutoConfig, AutoTokenizer

from utils import (
    math_equal,
    strip_string,
    load_instruction,
    compute_format_score,
)
from utils.parser import extract_answer


console = Console()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
namespace = uuid.NAMESPACE_DNS


class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()


class DatasetLoader:
    """Handles loading of different mathematical datasets"""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load_orz_math_57k(self, data_path: str) -> List[List[str]]:
        if self.data_path:
            data_path = self.data_path
        df = pd.read_json(data_path)
        questions = [item["value"] for item in df[0].tolist()]
        answers = [item["ground_truth"]["value"] for item in df[1].tolist()]
        return [questions, answers]

    def load_megascience(self, data_path: str) -> List[List[str]]:
        if self.data_path:
            data_path = self.data_path
        # df = pd.read_json(data_path)
        with open(data_path, "r") as f:
            df = json.load(f)
        questions = [item["problem"] for item in df]
        answers = [item["solution"] for item in df]
        return [questions, answers]

    def load_polaris_53k(self, data_path: str) -> List[List[str]]:
        if self.data_path:
            data_path = self.data_path
        with open(data_path, "r") as f:
            df = json.load(f)
        questions = [item["problem"] for item in df]
        answers = [item["answer"] for item in df]
        return [questions, answers]

    def load_dataset(
        self,
        name: str,
        tokenizer: AutoTokenizer,
        resume_from_checkpoint: Optional[str] = None,
    ) -> List[List[str]]:
        """Load dataset by name"""
        if name == "ORZ-MATH-57K":
            data_path = "dataset/orz_math_57k_collection/orz_math_57k_collection.json"
            dset = self.load_orz_math_57k(data_path)
        elif name == "MegaScience":
            data_path = "dataset/megascience/megascience.json"
            dset = self.load_megascience(data_path)
        elif name == "Polaris-53K":
            data_path = "dataset/polaris_53k/polaris_53k.json"
            dset = self.load_polaris_53k(data_path)
        else:
            raise ValueError(f'Dataset "{name}" not supported.')

        # Deduplicate problems if resuming from checkpoint directory
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            existing_latest_files = [
                f
                for f in os.listdir(resume_from_checkpoint)
                if f.startswith(f"rejection_sampling_{name}_") and f.endswith(".json")
            ]
            if len(existing_latest_files) > 0:
                latest_file = max(existing_latest_files)
                with open(os.path.join(resume_from_checkpoint, latest_file), "r") as f:
                    existing_results = json.load(f)
                existing_problems = set()
                for item in existing_results:
                    prompt = item["prompt"]
                    # Extract the problem statement from the prompt
                    problem = (
                        prompt.split("\n\nYou must write your answer strictly", 1)[0]
                        .split("<|im_start|>user\n", 1)[-1]
                        .strip()
                    )
                    existing_problems.add(problem)

                filtered_dset = [[], []]
                for i in range(len(dset[0])):
                    prob = dset[0][i]
                    if prob not in existing_problems:
                        filtered_dset[0].append(dset[0][i])
                        filtered_dset[1].append(dset[1][i])

                console.log(
                    f'Resuming from checkpoint "{latest_file}", filtered out {len(dset[0]) - len(filtered_dset[0])} existing problems.'
                )
                dset = filtered_dset
            else:
                console.log(
                    f'No existing results found in "{resume_from_checkpoint}". Starting fresh.'
                )
            console.log(f"Size of the dataset after filtering: {len(dset[0])}")
        return dset


class PromptBuilder:
    """Handles prompt construction for different models"""

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.instruction = load_instruction(self.args.instruction)
        console.log(f'Loaded instruction from "{self.args.instruction}".')
        console.rule()
        # syntax = Syntax(self.instruction, "html", theme="monokai", line_numbers=True)
        syntaxed_instruction = Syntax(
            self.instruction,
            "xml",
            theme="github-dark",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(syntaxed_instruction)
        console.rule()

    def build_prompt(self, question: str) -> str:
        """Build prompt for a given question"""
        # Determine system message based on model

        if self.args.structured_trace_distil:
            msgs = [
                {
                    "role": "user",
                    "content": question.strip() + "\n\n" + self.instruction,
                },
            ]
        else:
            # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
            msgs = [
                # {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": question.strip() + "\n\n" + self.instruction,
                },
            ]

        return self.tokenizer.apply_chat_template(
            # msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


class RejectionSampler:
    """Handles rejection sampling for mathematical problem solving."""

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # self.config = AutoConfig.from_pretrained(args.model_path)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if args.parallel_reasoning:
            console.log("[bold pink1]Parallel reasoning enabled.[/bold pink1]")
            stop_token_ids = [
                self.tokenizer.encode("</guideline>")[0],
                self.tokenizer.encode("</step>")[0],
            ]
        else:
            stop_token_ids = []

        # Build sampling parameters
        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "skip_special_tokens": False,
            "stop_token_ids": stop_token_ids,
            "no_stop_trim": True,
        }
        if args.temperature is not None:
            self.sampling_params["temperature"] = args.temperature
        if args.top_p is not None:
            self.sampling_params["top_p"] = args.top_p
        if args.top_k is not None:
            self.sampling_params["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            self.sampling_params["repetition_penalty"] = args.repetition_penalty

    def extract_boxed_answer(self, prediction: str) -> str:
        """Extract boxed prediction from full prediction."""
        answer = re.findall(r"\\boxed\{([^}]*)\}", prediction)
        if len(answer) > 0:
            answer = answer[0].strip()
        else:
            pos = prediction.find("</think>")
            answer = (
                prediction[pos + len("</think>") :].strip() if pos != -1 else prediction
            )
        return answer

    def generate_with_retry(self, prompts: List[str]) -> List[str]:
        """Generate responses with OOM handling."""
        while True:
            with torch.no_grad():
                loop = asyncio.get_event_loop()
                # generated = self.model.generate(prompts, self.sampling_params)
                generated = loop.run_until_complete(
                    self.model.async_generate(
                        prompt=prompts,
                        sampling_params=self.sampling_params,
                    ),
                )
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.model.flush_cache())
            return [r["text"] for r in generated]

    def postprocess_trajectory(self, trajectory: str) -> str:
        """Post-process the generated trajectory."""
        # Remove any text after the boxed answer
        trajectory = (
            trajectory.split("<|im_start|>assistant\n", 1)[-1]
            # .rsplit("<|im_end|>", 1)[0]
            .strip()
        )
        return trajectory


def initialize_model_and_tokenizer(args: argparse.Namespace):
    """Initialize model and tokenizer"""
    console.log(f"Loading model and tokenizer from {args.model_path}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left", use_fast=False
    )

    if hasattr(config, "max_position_embeddings"):
        console.log(f"Max position embeddings: {config.max_position_embeddings}")
    else:
        console.log(f"Max position embeddings: {config.model_max_length}")

    tp_size = args.tp_size if args.tp_size else torch.cuda.device_count()
    model = AsyncEngine(
        model_path=args.model_path,
        tp_size=tp_size,
        disable_overlap_schedule=True,
        dtype=torch.bfloat16,
        # log_level="info",
        mem_fraction_static=0.8,
    )

    return model, tokenizer


class BatchManager:
    def __init__(self, problems, prompts, answers, batch_size):
        self.problems = problems
        self.prompts = prompts
        self.answers = answers
        self.total = len(prompts)
        self.ptr = 0  # pointer to the next sample to fetch
        self.active = []  # current active batch (list of (prompt, answer, idx))
        self.batch_size = batch_size

    def fill_batch(self):
        """Fill the active batch up to batch_size"""
        while len(self.active) < self.batch_size and self.ptr < self.total:
            self.active.append(
                (
                    self.problems[self.ptr],
                    self.prompts[self.ptr],
                    self.answers[self.ptr],
                    self.ptr,
                )
            )
            self.ptr += 1

    def get_batch(self):
        """Return the current active batch"""
        return self.active

    def remove(self, solved_indices):
        """Remove solved samples from the active batch"""
        self.active = [x for x in self.active if x[3] not in solved_indices]


def rejection_sampling(
    args: argparse.Namespace,
    batch_manager: BatchManager,
    rejection_sampler: RejectionSampler,
):
    is_checked = False
    results = dict()
    save_results = dict()
    total_problems = len(batch_manager.prompts)
    num_discarded = 0
    num_trials = 0
    while True:
        num_trials += 1
        # Get dynamically filled batch
        batch_manager.fill_batch()
        batch = batch_manager.get_batch()
        if not batch:  # all samples finished
            break
        batch_problems, batch_prompts, batch_answers, batch_indices = zip(*batch)

        # process the records
        for i, global_idx in enumerate(batch_indices):
            if global_idx in results:
                results[global_idx]["sample_trial"] += 1
                continue
            reference = strip_string(batch_answers[i], skip_unit=False)
            results[global_idx] = {
                "id": str(uuid.uuid5(namespace, batch_problems[i])),
                "time": datetime.now().isoformat().rsplit(".")[0],
                "problem": batch_problems[i],
                "prompt": batch_prompts[i],
                "reference": reference,
                "trajectory": [],
                "boxed_answer": [],
                "sample_trial": 1,
            }

        # Sample the batch
        trajectory = rejection_sampler.generate_with_retry(list(batch_prompts))
        boxed_answers = [
            extract_answer(pred, args.dataset, use_last_number=True)
            for pred in trajectory
        ]

        if rejection_sampler.local_rank == 0 and args.debug and not is_checked:
            for i, global_idx in enumerate(batch_indices):
                if i < args.log_samples:
                    syntaxed_trajectory = Syntax(
                        trajectory[i],
                        "xml",
                        theme="github-dark",
                        line_numbers=True,
                        word_wrap=True,
                    )
                    console.rule()
                    console.print(f"Problem {global_idx} Prediction:")
                    console.print(syntaxed_trajectory)
                    console.rule()
                else:
                    break
            is_checked = True

        references = [results[idx]["reference"] for idx in batch_indices]
        boxed_answers_w_strikethrough = [
            (
                f"[strike bold]{ans}[/strike bold]"
                # if ans == references[i]
                if math_equal(prediction=ans, reference=references[i])
                else f"[bold red]{ans}[/bold red]"
            )
            for i, ans in enumerate(boxed_answers)
        ]
        references = [
            (
                f"[strike bold white]{ref}[/strike bold white]"
                if ref == boxed_answers[i]
                else f"[white]{ref}[/white]"
            )
            for i, ref in enumerate(references)
        ]
        try:
            console.log(f"Sampled answers: {boxed_answers_w_strikethrough}")
            # console.log(f"Refered answers: {references}")
        except Exception as _:
            print(f"Sampled answers: {boxed_answers_w_strikethrough}")
            # print(f"Refered answers: {references}")

        # rejection filtering (only leave those with correct boxed answers)
        filtered_trajectory = []
        newly_solved_problems = []
        references = [results[idx]["reference"] for idx in batch_indices]
        for global_idx, traj, boxed_pred, reference in zip(
            batch_indices, trajectory, boxed_answers, references
        ):
            if math_equal(prediction=boxed_pred, reference=reference):
                filtered_trajectory.append(traj)
                cleaned_traj = rejection_sampler.postprocess_trajectory(traj)
                if args.structured_trace_distil:
                    format_score = compute_format_score(cleaned_traj)
                    if format_score < 0.0:
                        continue

                results[global_idx]["trajectory"].append(cleaned_traj)
                results[global_idx]["boxed_answer"].append(boxed_pred)
                if (
                    len(results[global_idx]["trajectory"])
                    == args.max_trajectory_per_problem
                ):
                    newly_solved_problems.append(global_idx)

                    reference = results[global_idx]["reference"]
                    preds = results[global_idx]["boxed_answer"]
                    trajectory = results[global_idx]["trajectory"]

                    for idx in range(len(preds)):
                        results[global_idx]["boxed_answer"][idx] = (
                            r"\boxed{" + results[global_idx]["boxed_answer"][idx] + r"}"
                        )

                    full_preds_cleaned = [
                        traj.split("<|im_start|>assistant", 1)[-1].strip()
                        for traj in trajectory
                    ]
                    avg_format_score = sum(
                        [compute_format_score(pred) for pred in full_preds_cleaned]
                    ) / len(full_preds_cleaned)

                    avg_completed_tokens = sum(
                        [
                            len(rejection_sampler.tokenizer.encode(pred))
                            for pred in full_preds_cleaned
                        ]
                    ) / len(full_preds_cleaned)
                    input_tokens = len(
                        rejection_sampler.tokenizer.encode(
                            results[global_idx]["prompt"]
                        )
                    )

                    results[global_idx]["format_score"] = avg_format_score
                    results[global_idx]["input_tokens"] = input_tokens
                    results[global_idx]["completed_tokens"] = int(avg_completed_tokens)
                    results[global_idx]["num_correct_trajectory"] = len(
                        results[global_idx]["trajectory"]
                    )

                    if results[global_idx]["sample_trial"] >= args.min_sample_trial:
                        save_results[global_idx] = results.pop(global_idx)
                    else:
                        results.pop(global_idx)
                        num_discarded += 1

                    batch_manager.remove([global_idx])

        # Check for problems that reached max_sample_trial and should be removed
        problems_to_remove = []
        for global_idx in batch_indices:
            if (
                global_idx in results
                and results[global_idx]["sample_trial"] >= args.max_sample_trial
            ):
                if rejection_sampler.local_rank == 0:
                    if args.save_failed_case:
                        save_results[global_idx] = results.pop(global_idx)
                    else:
                        results.pop(global_idx)
                        num_discarded += 1
                else:
                    results.pop(global_idx)
                problems_to_remove.append(global_idx)

        if problems_to_remove:
            batch_manager.remove(problems_to_remove)

        # console.log(
        # f"\nNewly received {len(filtered_trajectory)} trajectories & solved {len(newly_solved_problems)} problems."
        # )

        # Save results periodically
        # 2025-09-12T08:56:23
        if len(save_results) > 0 and num_trials % args.save_interval == 0:
            date_suffix = datetime.now().strftime("%Y%m%dT%H%M%S")
            output_path = os.path.join(
                f"{args.output_dir}",
                f"rejection_sampling_{args.dataset}_{date_suffix}.json",
            )
            with open(output_path, "w") as f:
                list_of_results = [save_results[idx] for idx in save_results]
                json.dump(list_of_results, f, indent=4)

            args.num_saved = len(list_of_results)
            args.num_discarded = num_discarded
            with open(
                os.path.join(
                    f"{args.output_dir}",
                    "record.json",
                ),
                "w",
            ) as f:
                json.dump(vars(args), f, indent=4)

            console.log(
                f'Checkpointing {len(list_of_results)} results to "{output_path}".'
            )
            console.rule()

    # Save final results
    date_suffix = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = os.path.join(
        f"{args.output_dir}",
        f"rejection_sampling_{args.dataset}_{date_suffix}.json",
    )
    with open(output_path, "w") as f:
        list_of_results = [save_results[idx] for idx in save_results]
        json.dump(list_of_results, f, indent=4)

    args.num_discarded = num_discarded
    with open(
        os.path.join(
            f"{args.output_dir}",
            "args.json",
        ),
        "w",
    ) as f:
        json.dump(vars(args), f, indent=4)

    console.log(f'Checkpointing {len(list_of_results)} results to "{output_path}".')
    console.rule()

    # Calculate statistics
    solved_problems = len(save_results)
    total_samples = sum([save_results[idx]["sample_trial"] for idx in save_results])
    avg_format_score = (
        sum([save_results[idx].get("format_score", 0) for idx in save_results])
        / solved_problems
        if solved_problems > 0
        else 0
    )
    avg_input_tokens = (
        sum([save_results[idx].get("input_tokens", 0) for idx in save_results])
        / solved_problems
        if solved_problems > 0
        else 0
    )
    avg_completed_tokens = (
        sum([save_results[idx].get("completed_tokens", 0) for idx in save_results])
        / solved_problems
        if solved_problems > 0
        else 0
    )
    # accuracy = total_correct / total_problems if total_problems > 0 else 0
    samples_per_problem = total_samples / solved_problems if solved_problems > 0 else 0

    console.log(f"Total problems: {total_problems}")
    console.log(f"Total solved problems: {solved_problems}")
    console.log(f"Total accepted trajectories generated: {total_samples}")
    console.log(f"Average sampling trial per problem: {samples_per_problem:.1f}")
    console.log(
        f"Acceptance rate (problem): {solved_problems}/{total_problems} = {solved_problems / total_problems:.2f}"
    )
    console.log(
        f"Acceptance rate (trajectory): {total_samples}/{total_problems * args.max_sample_trial} = {total_samples / (total_problems * args.max_sample_trial):.2f}"
    )
    # console.log(f"Accuracy after rejection sampling: {accuracy:.3f}")
    console.log(f"Average format score: {avg_format_score:.3f}")
    console.log(f"Average input tokens: {avg_input_tokens:.1f}")
    console.log(f"Average completed tokens: {avg_completed_tokens:.1f}")


def main(args):
    sglang.set_default_backend("vllm")

    # Load model and tokenizer
    console.log(f'Employ model "{args.model_path}" to roll out trajectories.')
    model, tokenizer = initialize_model_and_tokenizer(args)

    # Prepare data pipeline
    dataset_loader = DatasetLoader()
    problems, answers = dataset_loader.load_dataset(
        args.dataset, tokenizer, args.resume_from_checkpoint
    )

    if args.max_problems is not None:
        problems = problems[: args.max_problems]
        answers = answers[: args.max_problems]

    console.log(f'Load {len(problems)} problems from "{args.dataset}".')
    prompt_builder = PromptBuilder(args, tokenizer)

    # Build prompts
    prompts = []
    for problem in problems:
        prompt = prompt_builder.build_prompt(problem)
        prompts.append(prompt)

    # Create dataset and dataloader
    batch_manager = BatchManager(problems, prompts, answers, args.batch_size)

    # Prepare output directory
    os.makedirs(f"{args.output_dir}", exist_ok=True)

    # Start rejection sampling
    rejection_sampler = RejectionSampler(args, model, tokenizer)

    rejection_sampling(args, batch_manager, rejection_sampler)
    console.log("[bold yellow]✅ Rejection sampling completed![/bold yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to evaluate (e.g., ORZ-MATH-57K)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="prompts/rejection_sampling_wt.txt",
        help="Instruction file for prompt construction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Directory to resume from checkpoint",
    )

    # Parallelism parameters
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallel size (defaults to available GPU count)",
    )

    # Generation parameters
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate",
    )
    parser.add_argument(
        "--log_samples",
        type=int,
        default=1,
        help="Log every N samples for debugging",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=30000,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=None, help="Repetition penalty"
    )

    # Batch processing
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--min_sample_trial",
        type=int,
        default=1,
        help="Minimum number of samples per problem",
    )
    parser.add_argument(
        "--max_sample_trial",
        type=int,
        default=8,
        help="Maximum number of samples per problem",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Number of trials between saving intermediate results",
    )
    parser.add_argument(
        "--save_failed_case",
        action="store_true",
        help="Whether to save failed cases",
    )
    parser.add_argument(
        "--parallel_reasoning",
        action="store_true",
        help="Enable parallel reasoning",
    )
    parser.add_argument(
        "--max_trajectory_per_problem",
        type=int,
        default=4,
        help="Max trajectories to keep per problem",
    )
    parser.add_argument(
        "--drop_useless", action="store_true", help="Drop empty answers"
    )
    parser.add_argument(
        "--structured_trace_distil",
        action="store_true",
        help="Enable structured trace distillation",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    main(args)
