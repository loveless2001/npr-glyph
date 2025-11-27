#!/usr/bin/env python3
"""
parquet2json.py - Convert Parquet files to JSON format
Usage: python parquet2json.py <input_file.parquet>
"""

import sys
import os
import pandas as pd
import argparse


def convert_parquet_to_json(input_path):
    """
    Convert a Parquet file to JSON format.
    
    Args:
        input_path (str): Path to the input Parquet file
    
    Returns:
        str: Path to the output JSON file
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if input file has .parquet extension
    if not input_path.lower().endswith('.parquet'):
        raise ValueError("Input file must have .parquet extension")
    
    # Generate output filename by replacing .parquet with .json
    output_path = os.path.splitext(input_path)[0] + '.json'
    
    try:
        # Read the Parquet file
        print(f"Reading Parquet file: {input_path}")
        df = pd.read_parquet(input_path)
        
        # Convert to JSON and save
        print(f"Converting to JSON: {output_path}")
        df.to_json(output_path, orient='records', indent=2)
        
        print(f"Conversion completed successfully!")
        print(f"Output file: {output_path}")
        print(f"Records converted: {len(df)}")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Error during conversion: {str(e)}")


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description='Convert Parquet files to JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parquet2json.py data.parquet
  python parquet2json.py /path/to/myfile.parquet
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input Parquet file'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        # Convert the file
        output_file = convert_parquet_to_json(args.input_file)
        
        if args.verbose:
            # Display file size information
            input_size = os.path.getsize(args.input_file)
            output_size = os.path.getsize(output_file)
            print(f"\nFile sizes:")
            print(f"  Input (Parquet): {input_size:,} bytes")
            print(f"  Output (JSON): {output_size:,} bytes")
            print(f"  Size ratio: {output_size/input_size:.2f}x")
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
