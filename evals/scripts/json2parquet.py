#!/usr/bin/env python3
"""
Convert JSON files to Parquet format.

Usage:
    python json2parquet.py input.json [output.parquet]

If output filename is not provided, it will use the input filename with .parquet extension.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd


def load_json_file(filepath):
    """Load JSON data from file, handling both single objects and arrays."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If it's a single object, wrap it in a list for DataFrame creation
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON file must contain an object or array of objects")

        return data
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)


def json_to_parquet(input_file, output_file=None):
    """Convert JSON file to Parquet format."""
    input_path = Path(input_file)

    # Generate output filename if not provided
    if output_file is None:
        output_file = input_path.with_suffix(".parquet")

    output_path = Path(output_file)

    print(f"Converting {input_path} to {output_path}")

    # Load JSON data
    data = load_json_file(input_path)

    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        if df.empty:
            print("Warning: Input JSON contains no data")
            return

        # Write to Parquet
        df.to_parquet(output_path, engine="pyarrow", index=False)

        print(f"Successfully converted {len(df)} records to {output_path}")
        print(f"Output file size: {output_path.stat().st_size:,} bytes")

    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python json2parquet.py data.json
    python json2parquet.py input.json output.parquet
    python json2parquet.py /path/to/data.json /path/to/output.parquet
        """,
    )

    parser.add_argument("input_file", help="Input JSON file path")

    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output Parquet file path (optional, defaults to input filename with .parquet extension)",
    )

    parser.add_argument("--version", action="version", version="json2parquet 1.0.0")

    # Check if required packages are available
    try:
        import pandas as pd
        import pyarrow
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("Please install required packages:")
        print("pip install pandas pyarrow")
        sys.exit(1)

    args = parser.parse_args()

    # Perform conversion
    json_to_parquet(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
