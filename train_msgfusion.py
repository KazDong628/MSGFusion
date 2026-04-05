#!/usr/bin/env python3
"""Entry point: MSGFusion IVIF training (run from repository root)."""

from experiments.train_msgfusion_ivif import run_msgfusion_training_loop


def main() -> None:
    run_msgfusion_training_loop()


if __name__ == "__main__":
    main()
