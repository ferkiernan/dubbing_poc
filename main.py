#!/usr/bin/env python3
"""Punto de entrada directo: python main.py run video.mp4 -o out.mp4 --voice es-f1"""

from dubbing_poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
