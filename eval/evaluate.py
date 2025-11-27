import asyncio
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from rich.console import Console
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, set_seed

from utils import (
    math_equal,
    strip_string,
    extract_answer,
    compute_format_score,
    parse_math_arena_answer,
    extract_math_arena_answer,
)

# Constants
DEFAULT_MAX_TOKENS = 30000
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_SAMPLES = 8
DEFAULT_PASS_K = 1
DEFAULT_MEM_FRACTION = 0.9

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.set_int_max_str_digits(1000000)

console = Console()


@dataclass
class TimingStats:
    """Statistics for timing and throughput metrics."""

    elapsed_time: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    total_tokens: int = 0
    total_samples: int = 0


class InferenceTimer:
    """Tracks inference timing and throughput with clean interface."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._total_tokens = 0
        self._total_samples = 0
        self._batch_times: List[float] = []

    def start(self) -> None:
        """Start the timer and log the start time."""
        self._start_time = time.time()

    def end(self) -> None:
        """End the timer."""
        self._end_time = time.time()

    def add_batch_stats(
        self, tokens: int, samples: int, batch_time: Optional[float] = None
    ) -> None:
        """Add statistics for a processed batch."""
        self._total_tokens += tokens
        self._total_samples += samples
        if batch_time is not None:
            self._batch_times.append(batch_time)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        current_time = self._end_time or time.time()
        return current_time - self._start_time

    @property
    def stats(self) -> TimingStats:
        """Calculate and return timing statistics."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return TimingStats()

        return TimingStats(
            elapsed_time=elapsed,
            tokens_per_second=self._total_tokens / elapsed,
            samples_per_second=self._total_samples / elapsed,
            total_tokens=self._total_tokens,
            total_samples=self._total_samples,
        )

    def print_progress(
        self, current_sample: Optional[int] = None, total_samples: Optional[int] = None
    ) -> None:
        """Print current progress with timing information."""
        elapsed = self.elapsed_time
        elapsed_str = self._format_time(elapsed)

        progress_str = ""
        if current_sample is not None and total_samples is not None:
            progress_str = f"[{current_sample}/{total_samples}] "

        console.log(f"⏱️ {progress_str}Elapsed: {elapsed_str}")

    def print_final_stats(self) -> None:
        """Print final timing and throughput statistics."""
        if self._start_time is None:
            return

        stats = self.stats
        elapsed_str = self._format_time(stats.elapsed_time)

        console.rule("🏁 Inference Complete")
        console.log(f"⏱️ Elapsed: {elapsed_str}")
        console.log(f"📊 Total tokens: {stats.total_tokens:,}")
        console.log(f"📊 Total samples: {stats.total_samples:,}")
        console.log(f"🚀 Token throughput: {stats.tokens_per_second:.2f} tokens/s")
        console.log(f"🚀 Sample throughput: {stats.samples_per_second:.2f} samples/s")

        if self._batch_times:
            avg_batch_time = np.mean(self._batch_times)
            console.log(f"⏳ Average batch time: {avg_batch_time:.2f}s")
        console.rule()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time duration in a human-readable way."""
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            return f"{seconds:.1f}s"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""

    model_path: str
    instruction: Optional[str] = None
    tp_size: Optional[int] = None
    dp_size: Optional[int] = None
    mem_fraction_static: Optional[float] = None

    # Generation parameters
    max_problems: Optional[int] = None
    log_samples: int = 1
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    parallel_reasoning: bool = False
    enable_thinking: bool = False

    # Batch processing
    batch_size: int = DEFAULT_BATCH_SIZE
    num_samples: int = DEFAULT_NUM_SAMPLES
    passk: int = DEFAULT_PASS_K

    # Task configuration
    tasks: List[str] = field(
        default_factory=lambda: [
            "AIME24",
            "AIME25",
            "AMC23",
            "MATH500",
            "HMMT_Feb_2025",
            "Minerva",
            "Olympiad",
            "GPQA",
            "MMLU",
            "BBEH",
            "ZebraLogic",
        ]
    )
    task_dir: str = "dataset/benchmark"

    # Output configuration
    output_dir: str = "output"
    save_alias: str = ""
    overwrite: bool = False
    apply_chat: bool = True
    debug: bool = False
    save_total_limit: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_gpu_config()
        self._set_defaults()

    def _validate_gpu_config(self):
        """Validate GPU configuration parameters."""
        num_gpus = torch.cuda.device_count()

        if self.dp_size < 1:
            raise ValueError(f"dp_size must be at least 1, got {self.dp_size}")

        if self.tp_size > num_gpus or self.tp_size < 1:
            raise ValueError(
                f"tp_size ({self.tp_size}) must be between 1 and {num_gpus}"
            )

    def _set_defaults(self):
        """Set default values based on available hardware."""
        num_gpus = torch.cuda.device_count()

        if not self.tp_size:
            self.tp_size = num_gpus
        if not self.dp_size:
            self.dp_size = num_gpus


def parse_arguments() -> EvaluationConfig:
    """Parse command line arguments and return configuration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Math evaluation script with parallel sampling"
    )

    # Model configuration
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--instruction", type=str, help="Instruction file for prompt construction"
    )
    parser.add_argument("--tp_size", type=int, help="Tensor parallel size")
    parser.add_argument("--dp_size", type=int, help="Number of parallel engines")
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=DEFAULT_MEM_FRACTION,
        help="Memory fraction per engine",
    )

    # Generation parameters
    parser.add_argument(
        "--max_problems", type=int, help="Maximum number of problems to evaluate"
    )
    parser.add_argument(
        "--log_samples", type=int, default=1, help="Log every N samples"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty")
    parser.add_argument(
        "--enable_thinking", action="store_true", help="Enable thinking"
    )
    parser.add_argument(
        "--parallel_reasoning", action="store_true", help="Enable parallel reasoning"
    )

    # Batch processing
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples per problem",
    )
    parser.add_argument(
        "--passk", type=int, default=DEFAULT_PASS_K, help="k value for pass@k"
    )

    # Task configuration
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[
            "AIME24",
            "AIME25",
            "AMC23",
            "MATH500",
            "HMMT_Feb_2025",
            "Minerva",
            "Olympiad",
            "GPQA",
            "MMLU",
            "BBEH",
            "ZebraLogic",
            "mix",
        ],
        help="Tasks to evaluate on",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default="dataset/benchmark",
        help="Directory containing task data",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--save_alias", type=str, default="", help="Alias to append to save path"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument(
        "--apply_chat", action="store_true", default=True, help="Apply chat template"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        help="Maximum number of result files to keep per dataset",
    )

    args = parser.parse_args()
    return EvaluationConfig(**vars(args))


class TokenizedDataset(Dataset):
    """Simple dataset wrapper for prompts and answers."""

    def __init__(self, prompts: List[str], answers: Union[List[str], List[List[str]]]):
        if len(prompts) != len(answers):
            raise ValueError("Prompts and answers must have the same length")
        self.prompts = prompts
        self.answers = answers

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return self.prompts[idx], self.answers[idx], idx

    def collate_fn(self, raw_batch):
        return zip(*raw_batch)


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Calculate pass@k metric using combinatorial formula."""
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


class DatasetLoader:
    """Handles loading of different mathematical datasets with unified interface."""

    def __init__(self, task_dir: str):
        self.task_dir = Path(task_dir)

    def load_dataset(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        """Load dataset by name with unified interface."""
        loader_map = {
            "AMC23": self._load_amc23,
            "AIME24": self._load_aime24,
            "AIME25": self._load_aime25,
            "MATH500": self._load_math500,
            "HMMT_Feb_2025": self._load_hmmt_feb_2025,
            "Minerva": self._load_minerva,
            "Olympiad": self._load_olympiad,
            "GPQA": self._load_gpqa,
            "MMLU": self._load_mmlu,
            "BBEH": self._load_bbeh,
            "ZebraLogic": self._load_zebra_logic,
            "mix": self._load_mix,
        }

        if dataset_name not in loader_map:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return loader_map[dataset_name]()

    def _load_json_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load JSON file with error handling."""
        try:
            with filepath.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")

    def _load_jsonl_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load JSONL file with error handling."""
        try:
            with filepath.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON line in {filepath}: {e}")

    def _load_mix(self) -> Tuple[List[str], List[str]]:
        """Load mix dataset."""
        filepath = self.task_dir / "mix" / "test.json"
        samples = self._load_json_file(filepath)
        return [s["question"] for s in samples], [str(s["answer"]) for s in samples]

    def _load_amc23(self) -> Tuple[List[str], List[str]]:
        """Load AMC23 dataset from json."""
        filepath = self.task_dir / "AMC23" / "amc23.json"
        samples = self._load_json_file(filepath)
        return [s["question"] for s in samples], [str(s["answer"]) for s in samples]

    def _load_aime24(self) -> Tuple[List[str], List[str]]:
        """Load AIME24 dataset from parquet."""
        df = pd.read_parquet(self.task_dir / "AIME24" / "aime24.parquet")
        questions = df["problem"].tolist()
        answers = [
            re.search(r"\\boxed\{(\d+)\}", text).group(1)
            for text in df["solution"].tolist()
        ]
        return questions, answers

    def _load_aime25(self) -> Tuple[List[str], List[str]]:
        """Load AIME25 dataset from multiple JSONL files."""
        questions, answers = [], []
        for filename in ["aime2025-I.jsonl", "aime2025-II.jsonl"]:
            filepath = self.task_dir / "AIME25" / filename
            samples = self._load_jsonl_file(filepath)
            questions.extend(s["question"] for s in samples)
            answers.extend(str(s["answer"]) for s in samples)
        return questions, answers

    def _load_math500(self) -> Tuple[List[str], List[str]]:
        """Load MATH500 dataset, returning only questions and answers."""
        filepath = self.task_dir / "MATH500" / "math500.json"
        samples = self._load_json_file(filepath)
        return [s["problem"] for s in samples], [str(s["answer"]) for s in samples]

    def _load_hmmt_feb_2025(self) -> Tuple[List[str], List[str]]:
        """Load HMMT Feb 2025 dataset from json."""
        filepath = self.task_dir / "HMMT_Feb_2025" / "hmmt_feb_2025.json"
        samples = self._load_json_file(filepath)
        return [s["problem"] for s in samples], [str(s["answer"]) for s in samples]

    def _load_minerva(self) -> Tuple[List[str], List[str]]:
        """Load Minerva dataset from JSONL."""
        filepath = self.task_dir / "Minerva" / "minerva.jsonl"
        samples = self._load_jsonl_file(filepath)
        return [s["question"] for s in samples], [str(s["answer"]) for s in samples]

    def _load_olympiad(self) -> Tuple[List[str], List[str]]:
        """Load Olympiad dataset from JSON."""
        filepath = self.task_dir / "Olympiad" / "olympiad.json"
        samples = self._load_json_file(filepath)
        questions, answers = [], []
        for sample in samples:
            questions.append(sample["problem"])
            if len(sample["answer"]) != 1:
                raise ValueError("Expecting single answer in Olympiad Bench")
            answers.append(str(sample["answer"][0]))
        return questions, answers

    def _load_gpqa(self) -> Tuple[List[str], List[str]]:
        """Load GPQA Diamond dataset from JSON."""
        filepath = self.task_dir / "GPQA_Diamond" / "gpqa_diamond_processed.json"
        samples = self._load_json_file(filepath)
        return [s["question"] for s in samples], [s["answer"] for s in samples]

    def _load_mmlu(self) -> Tuple[List[str], List[str]]:
        """Load MMLU-Pro dataset from parquet."""
        filepath = self.task_dir / "MMLU-Pro" / "mmlu_pro_test_processed.json"
        samples = self._load_json_file(filepath)
        return [s["question"] for s in samples], [s["answer"] for s in samples]

    def _load_bbeh(self) -> Tuple[List[str], List[str]]:
        """Load BBEH dataset from json."""
        filepath = self.task_dir / "BBEH" / "bbeh_processed.json"
        samples = self._load_json_file(filepath)
        return [s["question"] for s in samples], [s["answer"] for s in samples]

    def _load_zebra_logic(self) -> Tuple[List[str], List[str]]:
        """Load ZebraLogic dataset with formatted multiple choice questions."""
        filepath = self.task_dir / "ZebraLogic" / "zebra_logic.json"
        samples = self._load_json_file(filepath)
        questions, answers = [], []

        for sample in samples:
            puzzle = sample["puzzle"]
            question = sample["question"]
            choices = sample["choices"]
            answer = sample["answer"]

            if answer not in choices:
                raise ValueError("Answer must be in choices")

            # Format choices as A., B., etc.
            choices_str = "".join(
                f"{chr(65 + i)}. {choice} " for i, choice in enumerate(choices)
            )
            ordered_answer = chr(65 + choices.index(answer))

            full_question = (
                f"Puzzle: {puzzle}\n\n"
                f"Question: {question}\n\n"
                f"Choices: {choices_str}\n\n"
                f"Answer:"
            )
            questions.append(full_question)
            answers.append(ordered_answer)

        return questions, answers


class PromptBuilder:
    """Handles prompt construction with model-specific templates."""

    def __init__(self, config: EvaluationConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model_name = Path(config.model_path).name

    def build_prompt(self, question: str) -> str:
        """Build prompt for a given question with appropriate template."""
        if not self.config.apply_chat:
            return self._build_simple_prompt(question)

        system_prompt = self._get_system_prompt()
        instruction = self._load_instruction() if self.config.instruction else None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = question
        if instruction:
            user_content += f"\n\n{instruction}"

        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

    def _build_simple_prompt(self, question: str) -> str:
        """Build simple prompt without chat template."""
        instruction = "Please think step by step and in parallel. Put your final answer within \\boxed{}."
        return f"{instruction}\n\n{question}"

    def _get_system_prompt(self) -> Optional[str]:
        """Get model-specific system prompt."""
        if "Multiverse-32B" in self.model_name:
            return "Please think step by step and in parallel. Put your final answer within \\boxed{}."
        return None

    def _load_instruction(self) -> Optional[str]:
        """Load instruction from file if specified."""
        if not self.config.instruction:
            return None
        from utils import load_instruction

        return load_instruction(self.config.instruction)


class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    """Enhanced async engine with better signal handling and initialization."""

    def __init__(self, **kwargs):
        self.engine_id = kwargs.pop("engine_id", 0)
        self._need_reload = True

        # Safe signal handling for Ray workers
        with self._safe_signal_context():
            super().__init__(**kwargs)

    @staticmethod
    def _safe_signal_context():
        """Context manager for safe signal handling in Ray workers."""
        import signal
        import threading
        from contextlib import contextmanager

        @contextmanager
        def safe_signal_manager():
            original_signal = signal.signal

            def safe_signal_handler(signum, handler):
                if threading.current_thread() is threading.main_thread():
                    try:
                        return original_signal(signum, handler)
                    except ValueError:
                        return None
                return None

            signal.signal = safe_signal_handler
            try:
                yield
            finally:
                signal.signal = original_signal

        return safe_signal_manager()

    async def flush_cache(self):
        """Flush tokenizer cache."""
        return await self.tokenizer_manager.flush_cache()


@ray.remote(num_cpus=1)
class AsyncSglangEngine:
    """Ray actor wrapper for SGLang AsyncEngine with clean interface."""

    def __init__(
        self,
        engine_id: int,
        model_path: str,
        tp_size: int = 1,
        mem_fraction_static: float = 0.8,
        **kwargs,
    ):
        self.engine_id = engine_id
        self.model_path = model_path
        self.tp_size = tp_size
        self.mem_fraction_static = mem_fraction_static
        self.kwargs = kwargs
        self.engine: Optional[AsyncEngine] = None
        self._initialized = False

    async def init_engine(self) -> None:
        """Initialize the SGLang engine if not already done."""
        if self._initialized:
            return

        self.engine = AsyncEngine(
            model_path=self.model_path,
            tp_size=self.tp_size,
            mem_fraction_static=self.mem_fraction_static,
            disable_overlap_schedule=True,
            dtype=torch.bfloat16,
            engine_id=self.engine_id,
            **self.kwargs,
        )
        self._initialized = True

    async def async_generate(self, prompt, sampling_params):
        """Generate responses asynchronously."""
        if not self._initialized:
            await self.init_engine()
        return await self.engine.async_generate(
            prompt=prompt, sampling_params=sampling_params
        )

    async def flush_cache(self):
        """Flush the engine cache."""
        if self.engine:
            return await self.engine.flush_cache()

    def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        if self.engine:
            self.engine.shutdown()
            self.engine = None
            self._initialized = False


@dataclass
class SamplingParams:
    """Sampling parameters configuration."""

    max_new_tokens: int
    skip_special_tokens: bool = False
    stop_token_ids: List[int] = field(default_factory=list)
    no_stop_trim: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def create_sampling_params(config: EvaluationConfig, tokenizer) -> SamplingParams:
    """Create sampling parameters from configuration."""
    stop_tokens = []

    if "multiverse" in config.instruction:
        stop_tokens.extend(["</Goal>", "</Path>"])
    else:
        stop_tokens.extend(["</guideline>", "</step>"])

    if config.parallel_reasoning:
        console.log("[bold white]Parallel reasoning enabled.[/bold white]")
        stop_token_ids = [
            tokenizer.encode(stop_tokens[0])[0],
            tokenizer.encode(stop_tokens[1])[0],
        ]
    else:
        stop_token_ids = []

    return SamplingParams(
        max_new_tokens=config.max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
    )


class AsyncEngineManager:
    """Manager for multiple Ray-based SGLang engines"""

    def __init__(
        self,
        model_path: str,
        dp_size: int,
        tp_size: int = 1,
        mem_fraction_static: float = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.kwargs = kwargs

        # Calculate memory fraction
        num_gpus = torch.cuda.device_count()
        if mem_fraction_static is not None:
            # When using Ray DP with TP, reduce memory fraction to leave room for
            # Ray actor overhead and NCCL communication buffers
            if tp_size > 1:
                # Apply a safety factor for TP communication overhead
                self.mem_fraction_static = mem_fraction_static * 0.85
                console.log(
                    f"[white]Adjusting mem_fraction from {mem_fraction_static:.3f} to "
                    f"{self.mem_fraction_static:.3f} for TP overhead[/white]"
                )
            else:
                self.mem_fraction_static = mem_fraction_static
        else:
            if dp_size <= num_gpus:
                # When using TP within Ray actors, use more conservative memory
                self.mem_fraction_static = 0.75 if tp_size > 1 else 0.8
            else:
                engines_per_gpu = (dp_size + num_gpus - 1) // num_gpus
                safety_factor = 0.85
                self.mem_fraction_static = (0.95 / engines_per_gpu) * safety_factor
                self.mem_fraction_static = max(0.05, self.mem_fraction_static)

        console.log(
            f"Engine Manager: Creating {dp_size} engine(s) with memory fraction {self.mem_fraction_static:.3f}"
        )

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors for engines
        self.engines = []
        self._create_engines()

    def _create_engines(self):
        """Create Ray actor engines"""
        num_gpus = torch.cuda.device_count()

        for i in range(self.dp_size):
            # Create engine actor with GPU assignment
            if self.dp_size <= num_gpus:
                # Each engine gets its own GPU
                assert num_gpus // self.dp_size == self.tp_size, (
                    f"tp_size {self.tp_size} must evenly divide num_gpus {num_gpus} "
                    f"when dp_size {self.dp_size} <= num_gpus"
                )
                engine_actor = AsyncSglangEngine.options(
                    num_gpus=self.tp_size, name=f"sglang_engine_{i}"
                ).remote(
                    engine_id=i,
                    model_path=self.model_path,
                    tp_size=self.tp_size,
                    mem_fraction_static=self.mem_fraction_static,
                    **self.kwargs,
                )
            else:
                # Multiple engines share GPUs - use fractional allocation
                gpu_fraction = 1.0 / ((self.dp_size + num_gpus - 1) // num_gpus)
                engine_actor = AsyncSglangEngine.options(
                    num_gpus=gpu_fraction, name=f"sglang_engine_{i}"
                ).remote(
                    engine_id=i,
                    model_path=self.model_path,
                    tp_size=self.tp_size,
                    mem_fraction_static=self.mem_fraction_static,
                    **self.kwargs,
                )
            self.engines.append(engine_actor)

    async def init_engines(self):
        """Initialize all engines in parallel"""
        _ = [engine.init_engine.remote() for engine in self.engines]

    async def _ray_future_to_asyncio(self, ray_future):
        """Convert Ray future to asyncio future"""
        while True:
            try:
                result = ray.get(ray_future, timeout=0.1)
                return result
            except ray.exceptions.GetTimeoutError:
                await asyncio.sleep(0.1)

    def get_engine(self, index: int):
        """Get engine by index"""
        return self.engines[index % len(self.engines)]

    def get_all_engines(self):
        """Get all engines"""
        return self.engines

    def shutdown_all(self):
        """Shutdown all engines"""
        _ = [engine.shutdown.remote() for engine in self.engines]
        time.sleep(3)
        console.log("Engine Manager: All engines shut down")


class Evaluator:
    """Handles data parallel evaluation with Ray-based engines using clean interface."""

    def __init__(
        self, engine_manager: AsyncEngineManager, tokenizer, config: EvaluationConfig
    ):
        self.engine_manager = engine_manager
        self.model_config = AutoConfig.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.config = config
        self.dp_size = engine_manager.dp_size
        self.tp_size = engine_manager.tp_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.timer = InferenceTimer()
        self.sampling_params = create_sampling_params(config, tokenizer).to_dict()

    def split_data(
        self, prompts: List[str], answers: List[str]
    ) -> List[Tuple[List[str], List[str], List[int]]]:
        """Split data into chunks for each engine with balanced distribution."""
        n = len(prompts)
        chunk_size = (n + self.dp_size - 1) // self.dp_size

        return [
            (
                prompts[start_idx:end_idx],
                answers[start_idx:end_idx],
                list(range(start_idx, end_idx)),
            )
            for i in range(self.dp_size)
            for start_idx, end_idx in [(i * chunk_size, min((i + 1) * chunk_size, n))]
            if start_idx < n
        ]

    async def evaluate_with_engine(
        self,
        engine_id: int,
        engine_actor,
        prompts: List[str],
        answers: List[str],
        indices: List[int],
        dataset_name: str,
        status: Status,
        completed_problems: List[int],
        total_problems: int,
    ) -> Tuple[Dict[int, Dict], Dict[str, int]]:
        """Evaluate a subset of data with a Ray-based engine"""

        results = {}
        total_tokens = 0
        total_samples = 0

        try:
            # Create batches from this engine's chunk
            dataset = TokenizedDataset(prompts, answers)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )

            for batch in dataloader:
                batch_prompts, batch_answers, batch_local_indices = batch

                # Convert local indices to global indices
                batch_global_indices = [indices[idx] for idx in batch_local_indices]
                batch_prompts = list(batch_prompts)

                batch_results, elapsed_time = await self.evaluate_batch(
                    engine_id,
                    engine_actor,
                    batch_prompts,
                    batch_answers,
                    batch_global_indices,
                    dataset_name,
                )
                results.update(batch_results)

                # Count tokens and samples for this batch
                for global_idx in batch_global_indices:
                    if global_idx in batch_results:
                        result = batch_results[global_idx]
                        # Count tokens from all predictions
                        for pred in result["full_preds"]:
                            pred_cleaned = pred.split("<|im_start|>assistant", 1)[
                                -1
                            ].strip()
                            total_tokens += len(self.tokenizer.encode(pred_cleaned))
                        total_samples += len(result["full_preds"])

                # Update progress counter and status
                completed_problems[0] += len(batch_global_indices)
                if self.local_rank == 0:
                    status.update(
                        f"[bold white]Evaluating {dataset_name}({completed_problems[0]}/{total_problems})...[/bold white]"
                    )

                # Allow other coroutines to run
                console.log(
                    f"SGLang Engine {engine_id} (elapsed {self.timer._format_time(elapsed_time)}) processed batch {batch_global_indices}"
                )
                engine_actor.flush_cache.remote()

                await asyncio.sleep(0)

        except Exception as e:
            console.log(
                f"Ray Engine {engine_id} encountered error: {e}", style="bold red"
            )
            raise

        return results, {"total_tokens": total_tokens, "total_samples": total_samples}

    async def evaluate_batch(
        self,
        engine_id: int,
        engine_actor,
        batch_prompts: List[str],
        batch_answers: List[str],
        batch_indices: List[int],
        dataset_name: str,
    ) -> Tuple[Dict[int, Dict], float]:
        """Evaluate a batch using Ray engine with single loop for all samples"""
        results = {}
        start_time = time.time()

        # Initialize results
        for i, global_idx in enumerate(batch_indices):
            if dataset_name in ["GPQA", "MMLU", "BBEH"]:
                reference = batch_answers[i]
            elif dataset_name == "HMMT_Feb_2025":
                reference, _ = parse_math_arena_answer(
                    str(batch_answers[i]), list_answer="," in str(batch_answers[i])
                )
                reference = str(reference)
            else:
                reference = strip_string(batch_answers[i], skip_unit=False)

            results[global_idx] = {
                "time": datetime.now().isoformat().rsplit(".", 1)[0].strip(),
                "prompt": batch_prompts[i],
                "sample_size": self.config.num_samples,
                "full_preds": [],
                "boxed_preds": [],
                "format_scores": [],
                "reference": reference,
            }

        # Create all prompt-sample combinations for single batch processing
        all_prompts = []
        prompt_sample_mapping = []  # Maps result index to (batch_idx, sample_id)

        for sample_id in range(self.config.num_samples):
            for batch_idx in range(len(batch_prompts)):
                all_prompts.append(batch_prompts[batch_idx])
                prompt_sample_mapping.append((batch_idx, sample_id))

        # Generate all samples in a single call
        set_seed(0)  # Set base seed for reproducibility
        is_checked = False
        batch_total_tokens = 0

        # Call Ray engine for generation of all prompts at once
        generation_future = engine_actor.async_generate.remote(
            prompt=all_prompts, sampling_params=self.sampling_params
        )

        # Convert Ray future to result
        full_preds_response = await self._ray_future_to_asyncio(generation_future)
        all_full_preds = [r["text"] for r in full_preds_response]

        # Log wrapped response from ray subprocess with rich.syntax XML
        if not is_checked and self.config.debug:
            console.rule(f"[bold]🔍 Case Sanity Check (Engine {engine_id})[/bold]")
            syntaxed_response = Syntax(
                all_full_preds[0],
                "xml",
                theme="github-dark",
                line_numbers=True,
                word_wrap=True,
            )
            console.print(syntaxed_response)
            console.rule()
            is_checked = True

        # Process all predictions and organize by batch_idx and sample_id
        for pred_idx, (batch_idx, sample_id) in enumerate(prompt_sample_mapping):
            full_pred = all_full_preds[pred_idx]
            global_idx = batch_indices[batch_idx]

            # Extract boxed prediction based on dataset
            if dataset_name != "HMMT_Feb_2025":
                boxed_pred = extract_answer(
                    full_pred, dataset_name, use_last_number=True
                )
            else:
                boxed_pred = str(
                    extract_math_arena_answer(
                        full_pred,
                        False,
                        True,
                        list_answer="," in str(batch_answers[batch_idx]),
                    )[0]
                )

            # Count tokens for this prediction
            pred_tokens = len(self.tokenizer.encode(full_pred))
            batch_total_tokens += pred_tokens

            # Clean prediction and compute format score
            pred_cleaned = full_pred.split("<|im_start|>assistant", 1)[-1].strip()
            format_score = compute_format_score(pred_cleaned)

            # Store predictions in correct order
            results[global_idx]["full_preds"].append(full_pred)
            results[global_idx]["boxed_preds"].append(boxed_pred)
            results[global_idx]["format_scores"].append(format_score)

        # Calculate scores and finalize results
        for global_idx in results:
            correct_count = 0
            correct_but_unformatted_count = 0
            reference = results[global_idx]["reference"]
            preds = results[global_idx]["boxed_preds"]
            full_preds = results[global_idx]["full_preds"]
            full_preds_cleaned = [
                pred.split("<|im_start|>assistant", 1)[-1].strip()
                for pred in full_preds
            ]
            format_scores = results[global_idx]["format_scores"]
            for pred, format_score in zip(preds, format_scores):
                pred_cleaned = strip_string(pred, skip_unit=False)
                if isinstance(reference, list):
                    if any(
                        math_equal(prediction=pred_cleaned, reference=ref)
                        for ref in reference
                    ):
                        correct_count += 1
                        if format_score != 0:
                            correct_but_unformatted_count += 1
                else:
                    if math_equal(prediction=pred_cleaned, reference=reference):
                        correct_count += 1
                        if format_score != 0:
                            correct_but_unformatted_count += 1
            avg_format_score = sum(format_scores) / len(format_scores)
            results[global_idx]["correct_count"] = correct_count
            results[global_idx]["correct_but_unformatted_count"] = (
                correct_but_unformatted_count
            )
            passk_score = estimate_pass_at_k(
                self.config.num_samples, correct_count, self.config.passk
            )
            unformated_passk_score = estimate_pass_at_k(
                self.config.num_samples,
                correct_but_unformatted_count,
                self.config.passk,
            )
            passn_score = estimate_pass_at_k(
                self.config.num_samples, correct_count, self.config.num_samples
            )
            avg_completed_tokens = sum(
                [len(self.tokenizer.encode(pred)) for pred in full_preds_cleaned]
            ) / len(full_preds_cleaned)
            input_tokens = len(self.tokenizer.encode(results[global_idx]["prompt"]))
            results[global_idx][f"pass@{self.config.passk}"] = passk_score
            results[global_idx]["format_score"] = avg_format_score
            results[global_idx][f"unformatted_pass@{self.config.passk}"] = (
                unformated_passk_score
            )
            results[global_idx][f"pass@{self.config.num_samples}"] = passn_score
            results[global_idx]["input_tokens"] = input_tokens
            results[global_idx]["completed_tokens"] = int(avg_completed_tokens)

            if self.config.debug:
                console.log(
                    f"Problem {global_idx}: avg pass@{self.config.passk}: {passk_score:.3f}; avg format score: {avg_format_score:.3f}\n",
                    f"            avg unformatted pass@{self.config.passk}: {unformated_passk_score:.3f}; pass@{self.config.num_samples}: {passn_score:.3f}",
                )

        elapsed_time = time.time() - start_time

        return results, elapsed_time

    async def _ray_future_to_asyncio(self, ray_future):
        """Convert Ray future to asyncio future"""
        while True:
            try:
                result = ray.get(ray_future, timeout=0.1)
                return result
            except ray.exceptions.GetTimeoutError:
                await asyncio.sleep(0.1)

    async def async_evaluate(
        self,
        prompts: List[str],
        answers: List[str],
        dataset_name: str,
        status: Status,
    ) -> Dict[int, Dict]:
        """Evaluate using Ray-based data parallelism"""
        # Start timing
        if self.local_rank == 0:
            self.timer.start()

        # Split data across engines
        chunks = self.split_data(prompts, answers)
        engines = self.engine_manager.get_all_engines()

        # Create a shared progress counter
        completed_problems = [0]  # Use list for mutability in nested function

        # Create async tasks for each engine
        tasks = []
        for engine_id, (engine_actor, chunk) in enumerate(zip(engines, chunks)):
            if not chunk:  # Skip if no data for this engine
                continue

            chunk_prompts, chunk_answers, chunk_indices = chunk

            task = self.evaluate_with_engine(
                engine_id,
                engine_actor,
                chunk_prompts,
                chunk_answers,
                chunk_indices,
                dataset_name,
                status,
                completed_problems,
                len(prompts),
            )
            tasks.append(task)

        # Run all tasks concurrently and gather results
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        total_tokens = 0
        total_samples = 0
        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                console.log(f"Task {i} failed with error: {result}", style="bold red")
                raise result
            # result is a tuple (results_dict, stats_dict)
            results_dict, stats_dict = result
            results.update(results_dict)
            total_tokens += stats_dict.get("total_tokens", 0)
            total_samples += stats_dict.get("total_samples", 0)

        # End timing and print stats
        if self.local_rank == 0:
            self.timer.add_batch_stats(total_tokens, total_samples)
            self.timer.end()
            self.timer.print_final_stats()

        return results


def init_manager_and_tokenizer(
    config: EvaluationConfig,
) -> Union[AsyncEngineManager, AutoTokenizer]:
    """Initialize model and tokenizer based on configuration mode."""
    console.log(
        f"Loading model & tokenizer from [bold white]{config.model_path}[/bold white]"
    )

    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True, padding_side="left", use_fast=False
    )

    # Log context window size
    if hasattr(model_config, "max_position_embeddings"):
        console.log(f"Max context window: {model_config.max_position_embeddings}")
    else:
        console.log(f"Max context window: {model_config.model_max_length}")

    return create_engine_manager(config), tokenizer


def create_engine_manager(config: EvaluationConfig) -> AsyncEngineManager:
    """Create Ray-based engine manager."""
    num_gpus = torch.cuda.device_count()
    console.log(
        f"Using Ray mode with dp_size={config.dp_size}, tp_size={config.tp_size}, available GPUs={num_gpus}"
    )
    return AsyncEngineManager(
        model_path=config.model_path,
        dp_size=config.dp_size,
        tp_size=config.tp_size,
        mem_fraction_static=config.mem_fraction_static,
    )


def cleanup_old_results(
    output_dir: Path, dataset_name: str, save_total_limit: int
) -> None:
    """Remove old result files to maintain save_total_limit."""
    if save_total_limit is None:
        return

    # Find all result files for this dataset
    result_files = [
        (file_path, file_path.stat().st_mtime)
        for file_path in output_dir.glob(f"{dataset_name.lower()}_avg*.json")
        if file_path.is_file()
    ]

    # Sort by modification time (newest first) and delete excess files
    result_files.sort(key=lambda x: x[1], reverse=True)
    for file_path, _ in result_files[save_total_limit:]:
        try:
            file_path.unlink()
            console.log(f"Deleted old result file: {file_path.name}")
        except OSError as e:
            console.log(f"Failed to delete {file_path}: {e}", style="bold red")


def evaluate(
    dataset_name: str,
    config: EvaluationConfig,
    tokenizer,
    dataset_loader: DatasetLoader,
    prompt_builder: PromptBuilder,
    evaluator: Evaluator,
) -> Optional[Tuple[float, float, float, float]]:
    """Evaluate a single dataset using the specified evaluator mode."""

    # Setup output path
    if "checkpoint-" in config.model_path:
        simple_model_name = (
            f'{config.model_path.split("/")[-2]}-{config.model_path.split("/")[-1]}'
        )
    else:
        simple_model_name = Path(config.model_path).name
    out_dir = (
        Path(config.output_dir) / "eval" / f"{simple_model_name}{config.save_alias}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config_output_path = (
        out_dir / f"{dataset_name.lower()}_avg{config.num_samples}_config.json"
    )
    all_results_output_path = (
        out_dir / f"{dataset_name.lower()}_avg{config.num_samples}_all_results.json"
    )
    summarized_results_output_path = (
        out_dir
        / f"{dataset_name.lower()}_avg{config.num_samples}_summarized_results.json"
    )
    if not config.overwrite and all_results_output_path.exists():
        console.log(f'"{all_results_output_path}" already exists, skipping...')
        return None

    # Load and prepare data
    console.log(f"Loading [bold white]{dataset_name}[/bold white] benchmark...")
    questions, targets = dataset_loader.load_dataset(dataset_name)

    if config.max_problems is not None:
        questions = questions[: config.max_problems]
        targets = targets[: config.max_problems]

    # Build prompts
    console.log(f"Building prompts for {len(questions)} problems...")
    prompts = [prompt_builder.build_prompt(question) for question in questions]
    answers = [
        [str(t) for t in target] if isinstance(target, list) else str(target)
        for target in targets
    ]

    # Evaluate based on mode
    status = Status(
        f"[bold white]Evaluating {dataset_name}(0/{len(prompts)})...[/bold white]",
        console=console,
        speed=4.0,
        refresh_per_second=32.0,
    )
    status.start()
    try:
        results = run_evaluation(
            config,
            evaluator,
            prompts,
            answers,
            dataset_name,
            status,
        )
    finally:
        status.stop()

    # Calculate final scores and standard deviations
    passk_scores = [results[k][f"pass@{config.passk}"] for k in results]
    format_scores = [results[k]["format_score"] for k in results]
    unformatted_passk_scores = [
        results[k][f"unformatted_pass@{config.passk}"] for k in results
    ]
    passn_scores = [results[k][f"pass@{config.num_samples}"] for k in results]

    # Calculate overall averages
    final_passk_score = sum(passk_scores) / len(prompts)
    final_format_score = sum(format_scores) / len(prompts)
    final_unformatted_passk_score = sum(unformatted_passk_scores) / len(prompts)
    final_passn_score = sum(passn_scores) / len(prompts)

    # Calculate run-level averages
    passk_scores_run_groups = []
    format_scores_run_groups = []
    unformatted_passk_scores_run_groups = []
    for idx in range(0, config.num_samples):
        passk_scores_run_level = []
        format_scores_run_level = []
        unformatted_passk_scores_run_level = []
        for k in results:
            pred = results[k]["boxed_preds"][idx]
            reference = results[k]["reference"]
            format_score = results[k]["format_scores"][idx]
            if isinstance(reference, list):
                passk_score = (
                    1.0 if any([math_equal(pred, r) for r in reference]) else 0.0
                )
                unformatted_passk_score = (
                    1.0
                    if any([math_equal(pred, r) for r in reference])
                    and format_score != 0
                    else 0.0
                )
            else:
                passk_score = 1.0 if math_equal(pred, reference) else 0.0
                unformatted_passk_score = (
                    1.0 if math_equal(pred, reference) and format_score != 0 else 0.0
                )
            passk_scores_run_level.append(passk_score)
            format_scores_run_level.append(format_score)
            unformatted_passk_scores_run_level.append(unformatted_passk_score)
        passk_scores_run_groups.append(passk_scores_run_level)
        format_scores_run_groups.append(format_scores_run_level)
        unformatted_passk_scores_run_groups.append(unformatted_passk_scores_run_level)
    passk_scores_run_level = [
        sum(group) / len(group) for group in passk_scores_run_groups
    ]
    format_scores_run_level = [
        sum(group) / len(group) for group in format_scores_run_groups
    ]
    unformatted_passk_scores_run_level = [
        sum(group) / len(group) for group in unformatted_passk_scores_run_groups
    ]

    problem_passk_scores = []
    problem_format_scores = []
    problem_unformatted_passk_scores = []
    problem_passk_score_stds = []
    problem_format_score_stds = []
    problem_unformatted_passk_score_stds = []
    for k in results:
        passk_scores_problem_level = []
        format_scores_problem_level = []
        unformatted_passk_scores_problem_level = []
        for idx in range(0, config.num_samples):
            pred = results[k]["boxed_preds"][idx]
            reference = results[k]["reference"]
            format_score = results[k]["format_scores"][idx]
            if isinstance(reference, list):
                passk_score = (
                    1.0 if any([math_equal(pred, r) for r in reference]) else 0.0
                )
                unformatted_passk_score = (
                    1.0
                    if any([math_equal(pred, r) for r in reference])
                    and format_score != 0
                    else 0.0
                )
            else:
                passk_score = 1.0 if math_equal(pred, reference) else 0.0
                unformatted_passk_score = (
                    1.0 if math_equal(pred, reference) and format_score != 0 else 0.0
                )
            passk_scores_problem_level.append(passk_score)
            format_scores_problem_level.append(format_score)
            unformatted_passk_scores_problem_level.append(unformatted_passk_score)
        passk_score_problem_mean = sum(passk_scores_problem_level) / len(
            passk_scores_problem_level
        )
        passk_score_problem_std = (
            np.std(passk_scores_problem_level)
            if len(passk_scores_problem_level) > 1
            else 0.0
        )
        format_score_problem_mean = (
            sum(format_scores_problem_level) / len(format_scores_problem_level)
            if len(format_scores_problem_level) > 0
            else 0.0
        )
        format_score_problem_std = (
            np.std(format_scores_problem_level)
            if len(format_scores_problem_level) > 1
            else 0.0
        )
        unformatted_passk_score_problem_mean = sum(
            unformatted_passk_scores_problem_level
        ) / len(unformatted_passk_scores_problem_level)
        unformatted_passk_score_problem_std = (
            np.std(unformatted_passk_scores_problem_level)
            if len(unformatted_passk_scores_problem_level) > 1
            else 0.0
        )
        problem_passk_scores.append(passk_score_problem_mean)
        problem_format_scores.append(format_score_problem_mean)
        problem_unformatted_passk_scores.append(unformatted_passk_score_problem_mean)
        problem_passk_score_stds.append(passk_score_problem_std)
        problem_format_score_stds.append(format_score_problem_std)
        problem_unformatted_passk_score_stds.append(unformatted_passk_score_problem_std)

    # Calculate standard deviations
    run_passk_score_std = (
        np.std(passk_scores_run_level) if len(passk_scores_run_level) > 1 else 0.0
    )
    run_format_score_std = (
        np.std(format_scores_run_level) if len(format_scores_run_level) > 1 else 0.0
    )
    run_unformatted_passk_score_std = (
        np.std(unformatted_passk_scores_run_level)
        if len(unformatted_passk_scores_run_level) > 1
        else 0.0
    )
    problem_passk_score_std = (
        np.mean(problem_passk_score_stds) if len(problem_passk_score_stds) > 1 else 0.0
    )
    problem_format_score_std = (
        np.mean(problem_format_score_stds)
        if len(problem_format_score_stds) > 1
        else 0.0
    )
    problem_unformatted_passk_score_std = (
        np.mean(problem_unformatted_passk_score_stds)
        if len(problem_unformatted_passk_score_stds) > 1
        else 0.0
    )

    # Print final scores
    console.log(
        f"{Path(config.model_path).name} {dataset_name} pass@{config.passk}: {final_passk_score * 100:.2f}%"
    )
    console.log(
        f"{Path(config.model_path).name} {dataset_name} avg format score: {final_format_score:.2f}"
    )
    console.log(
        f"{Path(config.model_path).name} {dataset_name} unformatted pass@{config.passk}: {final_unformatted_passk_score * 100:.2f}%"
    )
    console.log(
        f"{Path(config.model_path).name} {dataset_name} pass@{config.num_samples}: {final_passn_score * 100:.2f}%"
    )

    # Save results
    with config_output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=4)
    console.log(f'Configuration saved to "{config_output_path}"')

    with all_results_output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    console.log(f'All results saved to "{all_results_output_path}"')

    summarized_results = {
        "model": Path(config.model_path).name,
        "dataset": dataset_name,
        "num_problems": len(prompts),
        "num_samples": config.num_samples,
        f"pass@{config.passk}": final_passk_score,
        "format_score": final_format_score,
        f"unformatted_pass@{config.passk}": final_unformatted_passk_score,
        f"pass@{config.num_samples}": final_passn_score,
        "run_passk_score_std": run_passk_score_std,
        "run_format_score_std": run_format_score_std,
        "run_unformatted_passk_score_std": run_unformatted_passk_score_std,
        "problem_passk_score_std": problem_passk_score_std,
        "problem_format_score_std": problem_format_score_std,
        "problem_unformatted_passk_score_std": problem_unformatted_passk_score_std,
    }
    with summarized_results_output_path.open("w", encoding="utf-8") as f:
        json.dump(summarized_results, f, ensure_ascii=False, indent=4)
    console.log(f'Summarized results saved to "{summarized_results_output_path}"')

    # Cleanup old results if specified
    if config.save_total_limit is not None:
        cleanup_old_results(out_dir, dataset_name, config.save_total_limit)

    return (
        final_passk_score,
        final_format_score,
        final_unformatted_passk_score,
        final_passn_score,
        run_passk_score_std,
        run_format_score_std,
        run_unformatted_passk_score_std,
        problem_passk_score_std,
        problem_format_score_std,
        problem_unformatted_passk_score_std,
    )


def run_evaluation(
    config: EvaluationConfig,
    evaluator: Evaluator,
    prompts: List[str],
    answers: List[str],
    dataset_name: str,
    status: Status,
) -> Dict[int, Dict]:
    """Run Ray-based evaluation."""

    async def init_and_evaluate():
        await evaluator.engine_manager.init_engines()
        return await evaluator.async_evaluate(prompts, answers, dataset_name, status)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(init_and_evaluate())

    engines = evaluator.engine_manager.get_all_engines()
    for engine in engines:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(engine.flush_cache.remote())

    return results


def main() -> None:
    """Main evaluation function with improved structure."""
    sglang.set_default_backend("vllm")

    # Parse configuration
    config = parse_arguments()

    # Initialize components
    manager, tokenizer = init_manager_and_tokenizer(config)
    dataset_loader = DatasetLoader(config.task_dir)
    prompt_builder = PromptBuilder(config, tokenizer)

    # Create appropriate evaluator
    evaluator = Evaluator(manager, tokenizer, config)

    # Prepare output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate each dataset
    passk_score_summary = {}
    format_score_summary = {}
    unformatted_passk_score_summary = {}
    passn_score_summary = {}
    run_passk_score_std_summary = {}
    run_format_score_std_summary = {}
    run_unformatted_passk_score_std_summary = {}
    problem_passk_score_std_summary = {}
    problem_format_score_std_summary = {}
    problem_unformatted_passk_score_std_summary = {}

    for dataset_name in config.tasks:
        result = evaluate(
            dataset_name,
            config,
            tokenizer,
            dataset_loader,
            prompt_builder,
            evaluator,
        )
        if result is not None:
            (
                passk_score,
                format_score,
                unformatted_passk_score,
                passn_score,
                run_passk_score_std,
                run_format_score_std,
                run_unformatted_passk_score_std,
                problem_passk_score_std,
                problem_format_score_std,
                problem_unformatted_passk_score_std,
            ) = result
            passk_score_summary[dataset_name] = passk_score
            format_score_summary[dataset_name] = format_score
            unformatted_passk_score_summary[dataset_name] = unformatted_passk_score
            passn_score_summary[dataset_name] = passn_score
            run_passk_score_std_summary[dataset_name] = run_passk_score_std
            run_format_score_std_summary[dataset_name] = run_format_score_std
            run_unformatted_passk_score_std_summary[dataset_name] = (
                run_unformatted_passk_score_std
            )
            problem_passk_score_std_summary[dataset_name] = problem_passk_score_std
            problem_format_score_std_summary[dataset_name] = problem_format_score_std
            problem_unformatted_passk_score_std_summary[dataset_name] = (
                problem_unformatted_passk_score_std
            )

    # Print final summary
    if passk_score_summary:
        print_final_summary(
            config,
            passk_score_summary,
            format_score_summary,
            unformatted_passk_score_summary,
            passn_score_summary,
            run_passk_score_std_summary,
            run_format_score_std_summary,
            run_unformatted_passk_score_std_summary,
            problem_passk_score_std_summary,
            problem_format_score_std_summary,
            problem_unformatted_passk_score_std_summary,
        )

    # Cleanup resources
    cleanup_resources(config, manager)


def print_final_summary(
    config: EvaluationConfig,
    passk_score_summary: Dict[str, float],
    format_score_summary: Dict[str, float],
    unformatted_passk_score_summary: Dict[str, float],
    passn_score_summary: Dict[str, float],
    run_passk_score_std_summary: Dict[str, float],
    run_format_score_std_summary: Dict[str, float],
    run_unformatted_passk_score_std_summary: Dict[str, float],
    problem_passk_score_std_summary: Dict[str, float],
    problem_format_score_std_summary: Dict[str, float],
    problem_unformatted_passk_score_std_summary: Dict[str, float],
) -> None:
    """Print final evaluation summary table."""
    console.rule()
    table = Table(title=f"Final Evaluation Summary")
    table.add_column("Benchmark", justify="left", style="cyan", no_wrap=True)
    table.add_column(f"Pass@{config.passk} (±)", justify="right", style="magenta")
    table.add_column("Format Score (±)", justify="right", style="green")
    table.add_column(
        f"Unformatted Pass@{config.passk} (±)", justify="right", style="yellow"
    )
    table.add_column(f"Pass@{config.num_samples}", justify="right", style="cyan")

    for dataset_name, passk_score in passk_score_summary.items():
        format_score = format_score_summary[dataset_name]
        unformatted_passk_score = unformatted_passk_score_summary[dataset_name]
        passn_score = passn_score_summary[dataset_name]
        run_passk_score_std = run_passk_score_std_summary[dataset_name]
        run_format_score_std = run_format_score_std_summary[dataset_name]
        run_unformatted_passk_score_std = run_unformatted_passk_score_std_summary[
            dataset_name
        ]
        problem_passk_score_std = problem_passk_score_std_summary[dataset_name]
        problem_format_score_std = problem_format_score_std_summary[dataset_name]
        problem_unformatted_passk_score_std = (
            problem_unformatted_passk_score_std_summary[dataset_name]
        )
        table.add_row(
            dataset_name,
            f"{passk_score * 100:.2f}%\n(inter-run std: ±{run_passk_score_std * 100:.2f}%)\n(intra-problem std: ±{problem_passk_score_std * 100:.2f}%)",
            f"{format_score:.2f}\n(inter-run std: ±{run_format_score_std:.2f})\n(intra-problem std: ±{problem_format_score_std:.2f})",
            f"{unformatted_passk_score * 100:.2f}%\n(inter-run std: ±{run_unformatted_passk_score_std * 100:.2f}%)\n(intra-problem std: ±{problem_unformatted_passk_score_std * 100:.2f}%)",
            f"{passn_score * 100:.2f}%\n-\n-",
        )

    console.print(table)
    console.rule()


def cleanup_resources(config: EvaluationConfig, manager: AsyncEngineManager) -> None:
    """Cleanup model resources based on mode."""
    try:
        manager.shutdown_all()
    except Exception as e:
        console.log(f"Error shutting down engines: {e}", style="bold red")

    try:
        ray.shutdown()
    except Exception as e:
        console.log(f"Error shutting down Ray: {e}", style="bold red")


if __name__ == "__main__":
    main()
