#!/usr/bin/env python3
"""
Export stacks.yaml to stacks.json

Small utility to export the YAML metadata into a JSON file for easier consumption by JS tools or other frontends.
"""
import json
import sys
from pathlib import Path
import yaml


def main():
    repo_root = Path(__file__).resolve().parent.parent
    stacks_yaml = repo_root / 'templateheaven' / 'data' / 'stacks.yaml'
    stacks_json = repo_root / 'templateheaven' / 'data' / 'stacks.json'

    if not stacks_yaml.exists():
        print(f"Could not find stacks.yaml at {stacks_yaml}")
        sys.exit(1)

    with open(stacks_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    with open(stacks_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully wrote {stacks_json}")


if __name__ == '__main__':
    main()
