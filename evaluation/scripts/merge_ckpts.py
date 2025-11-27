#!/usr/bin/env python3
"""
Merge checkpoint files by reading JSON data, deduplicating by ID,
and sorting by time.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console

console = Console()


def read_directory_files(directory: str) -> List[Path]:
    """Read all JSON file paths from the given directory."""
    dir_path = Path(directory)

    if not dir_path.exists():
        console.print(f"[red]Error: Directory '{directory}' does not exist[/red]")
        sys.exit(1)

    if not dir_path.is_dir():
        console.print(f"[red]Error: '{directory}' is not a directory[/red]")
        sys.exit(1)

    json_files = list(dir_path.glob("*.json"))

    if not json_files:
        console.print(f"[yellow]Warning: No JSON files found in '{directory}'[/yellow]")
    else:
        console.print(f"[green]Found {len(json_files)} JSON file(s)[/green]")

    return json_files


def read_and_merge_files(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """Read JSON files and merge all data samples together."""
    all_samples = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Handle both single dict and list of dicts
                if isinstance(data, list):
                    all_samples.extend(data)
                elif isinstance(data, dict):
                    all_samples.append(data)

            console.print(f"[blue]✓[/blue] Read: {file_path.name}")
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Failed to parse {file_path.name}: {e}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Error reading {file_path.name}: {e}[/red]")

    console.print(f"[green]Total samples collected: {len(all_samples)}[/green]")
    return all_samples


def deduplicate_by_id(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate data samples by the 'id' key."""
    seen_ids = {}

    for sample in samples:
        sample_id = sample.get("id")
        if sample_id is not None:
            if sample_id not in seen_ids:
                seen_ids[sample_id] = sample

    deduplicated = list(seen_ids.values())
    console.print(
        f"[green]After deduplication: {len(deduplicated)} unique samples[/green]"
    )

    return deduplicated


def sort_by_time(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort data samples by the 'time' key in ascending order."""
    # Separate samples with and without 'time' key
    with_time = [s for s in samples if "time" in s]
    without_time = [s for s in samples if "time" not in s]

    # Sort samples that have 'time' key
    sorted_samples = sorted(with_time, key=lambda x: x["time"])

    # Append samples without 'time' at the end
    sorted_samples.extend(without_time)

    if without_time:
        console.print(
            f"[yellow]Note: {len(without_time)} sample(s) without 'time' key placed at end[/yellow]"
        )

    console.print(f"[green]Samples sorted by time (ascending)[/green]")
    return sorted_samples


def save_merged_data(samples: List[Dict[str, Any]], directory: str) -> Path:
    """Save the merged and processed data to a JSON file."""
    output_path = Path(directory) / "merged_ckpt.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    return output_path


def main():
    """Main function to orchestrate the merge process."""
    if len(sys.argv) != 2:
        console.print("[red]Usage: python merge_ckpts.py <directory_path>[/red]")
        sys.exit(1)

    directory = sys.argv[1]

    console.print(f"\n[bold cyan]Starting checkpoint merge process[/bold cyan]")
    console.print(f"[cyan]Directory: {directory}[/cyan]\n")

    # Step 1: Read file paths
    file_paths = read_directory_files(directory)

    if not file_paths:
        return

    # Step 2: Read and merge files
    console.print(f"\n[bold]Reading files...[/bold]")
    all_samples = read_and_merge_files(file_paths)

    if not all_samples:
        console.print("[yellow]No data samples to process[/yellow]")
        return

    # Step 3: Deduplicate by ID
    console.print(f"\n[bold]Deduplicating by 'id'...[/bold]")
    unique_samples = deduplicate_by_id(all_samples)

    # Step 4: Sort by time
    console.print(f"\n[bold]Sorting by 'time'...[/bold]")
    sorted_samples = sort_by_time(unique_samples)

    # Step 5: Save to file
    console.print(f"\n[bold]Saving merged data...[/bold]")
    output_path = save_merged_data(sorted_samples, directory)

    # Final summary
    console.print(f"\n[bold green]✓ Success![/bold green]")
    console.print(
        f"[green]Merged file saved to:[/green] [bold]{output_path.absolute()}[/bold]"
    )
    console.print(f"[green]Total samples in output:[/green] {len(sorted_samples)}\n")


if __name__ == "__main__":
    main()
