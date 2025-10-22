# src/ai_agent/cli.py
import argparse
import sys

def run_ui():
    from ai_agent.ui.app import launch
    launch()

def main():
    parser = argparse.ArgumentParser(description="AI Agent CLI")
    # default command is 'ui' (only option for now)
    parser.add_argument(
        "mode",
        choices=["ui"],
        help="Mode to run (currently only 'ui' is supported)."
    )
    args = parser.parse_args()

    if args.mode == "ui":
        run_ui()
    else:
        sys.exit(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()
