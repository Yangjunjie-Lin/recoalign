#!/usr/bin/env python3
"""Thin wrapper around the package table builder."""

import sys

from recoalign.cli import main

raise SystemExit(main(["build-table", *sys.argv[1:]]))
