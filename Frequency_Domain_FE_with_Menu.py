"""Compatibility entry point. See --help for explicit data and label inputs."""
import sys
from fault_diagnosis import main

if __name__ == "__main__":
    raise SystemExit(main(["extract", "--domain", "frequency"] + sys.argv[1:]))
