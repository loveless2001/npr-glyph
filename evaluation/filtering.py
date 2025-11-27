"""
Script for parallel logic scoring and filtering of
math reasoning trajectories using LLM-as-a-Judge.
"""

import os
import re
import json
import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import sglang
import sglang.srt.entrypoints.engine
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.status import Status
from transformers import AutoTokenizer, set_seed

from utils import load_protocol, extract_answer, strip_string

console = Console()

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        self.engine_id = kwargs.get("engine_id", 0)
        kwargs.pop("engine_id", None)
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()


class Evaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.max_problems = args.max_problems
        self.repeat_times = args.repeat_times if args.repeat_times else 1
        self.log_samples = args.log_samples if args.log_samples else 1
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def extract_score_from_response(self, response: str) -> Optional[float]:
        """Extract score from LLM-as-a-Judge response."""
        # Look for \boxed{score} pattern
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            score_text = boxed_match.group(1).strip()
            try:
                # Handle different score formats
                if "/" in score_text:
                    # Handle fractional scores like "4/5"
                    numerator, denominator = score_text.split("/")
                    return float(numerator) / float(denominator)
                else:
                    # Handle decimal scores like "0.8" or "4"
                    return float(score_text)
            except ValueError:
                pass

        # Fallback: look for explicit score patterns
        score_patterns = [
            r"[Ss]core[:\s]+([0-9]*\.?[0-9]+)",
            r"[Rr]ating[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*(?:/\s*[0-9]+)?\s*(?:out of|/)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    async def evaluate_sample(
        self,
        model: AsyncEngine,
        prompt_builder: "PromptBuilder",
        sample: Dict[str, Any],
        sample_idx: int,
    ) -> Dict[str, Any]:
        """Evaluate a single sample using LLM-as-a-Judge."""

        # Build evaluation prompt using the protocol
        evaluation_content = self._build_evaluation_content(sample)
        evaluation_prompt = prompt_builder.build_prompt(evaluation_content)

        # Generate response
        sampling_params = {
            "max_new_tokens": self.args.max_new_tokens,
            "skip_special_tokens": False,
            "stop_token_ids": [],
            "no_stop_trim": True,
        }

        # Add optional sampling parameters
        if self.args.temperature is not None:
            sampling_params["temperature"] = self.args.temperature
        if self.args.top_p is not None:
            sampling_params["top_p"] = self.args.top_p
        if self.args.top_k is not None:
            sampling_params["top_k"] = self.args.top_k
        if self.args.repetition_penalty is not None:
            sampling_params["repetition_penalty"] = self.args.repetition_penalty

        # Repeat evaluation multiple times if specified
        all_responses = []
        all_scores = []

        for repeat_idx in range(self.repeat_times):
            set_seed(42 + repeat_idx)  # Different seed for each repeat

            response = await model.async_generate(
                prompt=[evaluation_prompt], sampling_params=sampling_params
            )

            response_text = response[0]["text"]
            all_responses.append(response_text)

            # Extract score from response
            score = self.extract_score_from_response(response_text[-512:])
            all_scores.append(score)

            # Flush cache after each generation
            await model.flush_cache()

        # Calculate average score (excluding None values)
        valid_scores = [s for s in all_scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        # Log sample if debug mode
        if self.local_rank == 0 and sample_idx % self.log_samples == 0:
            console.rule(f"[bold]📊 Sample {sample_idx} Evaluation[/bold]")
            if self.repeat_times == 1:
                syntaxed_response = Syntax(
                    all_responses[0],
                    "xml",
                    theme="github-dark",
                    line_numbers=True,
                    word_wrap=True,
                )
                console.print(syntaxed_response)
            console.log(f"Score: {avg_score}")
            console.rule()

        return {
            "sample_idx": sample_idx,
            "timestamp": datetime.now().isoformat().rsplit(".", 1)[0].strip(),
            "original_sample": sample,
            "evaluation_prompt": evaluation_prompt,
            "responses": all_responses,
            "scores": all_scores,
            "avg_score": avg_score,
            "repeat_times": self.repeat_times,
        }

    def _build_evaluation_content(self, sample: Dict[str, Any]) -> str:
        """Build content for LLM-as-a-Judge evaluation."""
        # Extract question and response from sample
        question = sample.get("prompt")
        response = sample.get("trajectory")[0]

        # content = f"""Question: {question}

        # Response to evaluate: {response}"""
        content = f"Parallel solution to evaluate: {response}"

        return content


class PromptBuilder:
    """Handles prompt construction for model."""

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.model_name = os.path.basename(self.args.model_path)

    def build_prompt(self, content: str) -> str:
        """Build prompt for a given response generated by a language model."""
        if not self.args.apply_chat:
            inst = "Please think and evaluate step by step according to the given evaluation protocol. Put your final score within \\boxed{}."
            return f"{inst}\n\n{content}"

        # Determine system message based on model
        system_prompt = "Please think and evaluate step by step according to the given evaluation protocol and to-be-evaluated question and trajectory. Put your scoring results of solution within \\boxed{}."

        protocol = load_protocol(self.args.protocol) if self.args.protocol else None

        msgs = []

        # if system_prompt:
        # msgs.append({"role": "system", "content": system_prompt})
        if protocol:
            msgs.append(
                {
                    "role": "user",
                    "content": content + "\n\n" + protocol + "\n\n" + system_prompt,
                }
            )
        else:
            msgs.append({"role": "user", "content": content})

        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )


def load_dataset(file_path: str):
    """Load dataset from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Math evaluation script with parallel sampling"
    )

    # Model configuration
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallel size for 'tp' mode (defaults to available GPU count)",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="Path to the evaluation protocol file",
    )

    # Workflow configuration
    parser.add_argument(
        "--workflow",
        choices=[
            "all-in-one",
        ],
        default="all-in-one",
        help="Workflow to use for evaluation/filtering",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/filtered",
        help="Directory to save the filtered dataset",
    )

    # Generation parameters
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to filter",
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=None,
        help="Number of times to repeat evaluation or filtering for each problem",
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

    parser.add_argument(
        "--apply_chat", action="store_true", default=True, help="Apply chat template"
    )

    args = parser.parse_args()
    return args


def initialize_model_and_tokenizer(args: argparse.Namespace):
    """Initialize model and tokenizer"""
    console.log(f"Loading model or tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tp_size = args.tp_size if args.tp_size else torch.cuda.device_count()
    console.log(f"Using Tensor Parallelism with tp_size={tp_size}")

    model = AsyncEngine(
        model_path=args.model_path,
        tp_size=tp_size,
        disable_overlap_schedule=True,
        dtype=torch.bfloat16,
        # log_level="info",
        mem_fraction_static=0.8,
    )
    return model, tokenizer


async def evaluate_dataset_async(
    model: AsyncEngine,
    tokenizer,
    evaluator: Evaluator,
    prompt_builder: PromptBuilder,
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Asynchronously evaluate the entire dataset."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Limit dataset size if specified
    if args.max_problems and len(dataset) > args.max_problems:
        dataset = dataset[: args.max_problems]
        if local_rank == 0:
            console.log(f"Limited dataset to {len(dataset)} samples")

    results = []

    with Status(
        f"[bold white]Evaluating {len(dataset)} samples with LLM-as-a-Judge...[/bold white]",
        console=console,
        speed=2.0,
        refresh_per_second=16.0,
    ):
        start_time = time.time()

        for sample_idx, sample in enumerate(dataset):
            result = await evaluator.evaluate_sample(
                model, prompt_builder, sample, sample_idx
            )
            results.append(result)

            # Log progress & save periodically
            if local_rank == 0 and (sample_idx + 1) % max(1, len(dataset) // 1000) == 0:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = Path(args.model_path).name
                output_filename = f"filtered_{model_name}_{timestamp}.json"
                output_path = output_dir / output_filename
                elapsed = time.time() - start_time
                progress = (sample_idx + 1) / len(dataset)
                eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                console.log(
                    f"Progress: {sample_idx + 1}/{len(dataset)} ({progress*100:.1f}%) - "
                    f"ETA: {eta/60:.1f}m"
                )
                save_results(results, output_path, args)

    if local_rank == 0:
        elapsed = time.time() - start_time
        console.log(f"✅ Evaluation completed in {elapsed/60:.2f} minutes")

    return results


def save_results(
    results: List[Dict[str, Any]], output_path: Path, args: argparse.Namespace
) -> None:
    """Save evaluation results to file."""
    # Calculate summary statistics
    valid_scores = [r["avg_score"] for r in results if r["avg_score"] is not None]
    summary = {
        "total_samples": len(results),
        "successful_evaluations": len(valid_scores),
        "failed_evaluations": len(results) - len(valid_scores),
        "average_score": sum(valid_scores) / len(valid_scores)
        if valid_scores
        else None,
        "min_score": min(valid_scores) if valid_scores else None,
        "max_score": max(valid_scores) if valid_scores else None,
        "evaluation_params": {
            "model_path": args.model_path,
            "protocol": args.protocol,
            "repeat_times": args.repeat_times,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }

    output_data = {"summary": summary, "results": results}

    # Save results
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    console.log(f"📁 Results saved to {output_path}")

    # Print summary
    console.rule("[bold]📊 Evaluation Summary[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Samples", str(summary["total_samples"]))
    table.add_row("Successful Evaluations", str(summary["successful_evaluations"]))
    table.add_row("Failed Evaluations", str(summary["failed_evaluations"]))
    if summary["average_score"] is not None:
        table.add_row("Average Score", f"{summary['average_score']:.3f}")
        table.add_row(
            "Score Range", f"{summary['min_score']:.3f} - {summary['max_score']:.3f}"
        )

    console.print(table)
    console.rule()


def main():
    sglang.set_default_backend("vllm")

    # Parse arguments
    args = parse_arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize components based on mode
    model, tokenizer = initialize_model_and_tokenizer(args)
    prompt_builder = PromptBuilder(args, tokenizer)
    dataset = load_dataset(args.data_path)

    if local_rank == 0:
        console.log(f"Loaded {len(dataset)} samples from {args.data_path}")

    evaluator = Evaluator(args)

    # Setup output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    output_filename = f"filtered_{model_name}_{timestamp}.json"
    output_path = output_dir / output_filename

    set_seed(42)

    # Run evaluation
    async def run_evaluation():
        return await evaluate_dataset_async(
            model, tokenizer, evaluator, prompt_builder, dataset, args
        )

    # Execute evaluation
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(run_evaluation())

    # Save results
    if local_rank == 0:
        save_results(results, output_path, args)

    # Cleanup
    if local_rank == 0:
        console.log("🔧 Shutting down model...")
    model.shutdown()


if __name__ == "__main__":
    main()
