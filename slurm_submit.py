#!/usr/bin/env python3

"""
SLURM Job Submission Helper Script.

This script is designed to simplify the process of submitting jobs to the SLURM job scheduler.
It works as a wrapper around the `sbatch` command, allowing users to pass command-line flags directly
to an underlying shell script. Here's how it works:

1. Parse SLURM-specific command-line arguments, such as --partition, --gres, --mem, and --time.
2. Parse the main script to be executed along with its arguments.
3. Create a temporary SLURM batch script that:
    - Handles potential termination signals and ensures the script is moved to a .tmp_scripts directory upon completion or failure.
    - Includes the provided SLURM-specific arguments.
    - Calls the main script with its arguments.
4. Submits the temporary SLURM batch script using `sbatch`.

This way, users can submit jobs with custom parameters without manually creating a new SLURM batch script each time.

Usage:
    python3 this_script.py --partition=<partition_name> --mem=<memory> <path_to_script> <script_args>
"""


import argparse
import os
import re
import subprocess


def sanitize_filename(filename: str) -> str:
    # Remove invalid filename characters
    sanitized = re.sub(r"[^\w\-_\. ]", "_", filename)

    # Truncate to a reasonable length if necessary
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def main():
    parser = argparse.ArgumentParser(description="SLURM job submission helper.")
    parser.add_argument("--partition", help="SLURM partition.")
    parser.add_argument("--gres", help="SLURM generic resources.")
    parser.add_argument("--mem", help="SLURM memory requirement.")
    parser.add_argument("--time", help="SLURM time constraint.")
    parser.add_argument("script", help="The script to be executed.")
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments for the script."
    )

    args = parser.parse_args()

    # Ensure .tmp_scripts directory exists
    os.makedirs(".tmp_scripts", exist_ok=True)

    # Create the temporary batch script in the original directory
    tmp_filename = sanitize_filename(
        "tmp_" + "_".join([args.script] + args.script_args) + ".sh"
    )
    tmp_filepath = os.path.join(os.getcwd(), tmp_filename)

    with open(tmp_filepath, "w") as tmp:
        tmp.write(f"#!/bin/bash\n")
        # Trap both SIGTERM and EXIT signals to ensure script is moved to tmp folder
        if args.partition:
            tmp.write(f"#SBATCH --partition={args.partition}\n")
        if args.gres:
            tmp.write(f"#SBATCH --gres={args.gres}\n")
        if args.mem:
            tmp.write(f"#SBATCH --mem={args.mem}\n")
        if args.time:
            tmp.write(f"#SBATCH --time={args.time}\n")
        tmp.write(f"trap 'mv $0 ./.tmp_scripts/' SIGTERM EXIT\n")
        tmp.write(f"source $HOME/.bashrc\n")
        tmp.write(f"conda activate tp\n")
        # Add call to the desired script with its arguments
        script_args_str = " ".join(args.script_args)
        if args.script.endswith(".py"):
            tmp.write(f"python {args.script} {script_args_str}\n")
        else:
            tmp.write(f"bash {args.script} {script_args_str}\n")

    # Execute the sbatch command
    subprocess.run(["sbatch", tmp_filepath])


if __name__ == "__main__":
    main()
