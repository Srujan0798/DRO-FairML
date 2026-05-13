#!/usr/bin/env python3
"""Stop all running experiment processes."""
import subprocess
import sys

print("Stopping all experiment processes...")
result = subprocess.run(["pkill", "-f", "run_experiments.py"], capture_output=True)
if result.returncode == 0:
    print("Stopped.")
else:
    print("No processes found or already stopped.")
    
print("\nRemaining Python processes:")
subprocess.run(["ps", "aux", "|", "grep", "run_experiments", "|", "grep", "-v", "grep"], shell=True)
