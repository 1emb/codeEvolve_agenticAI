#!/usr/bin/env python
"""Shared helpers for the graph colouring problem builders: the table of best-known colour
counts, graph download from COLOR02, and the yaml writer. Imported by make_problem_fixedk.py,
calibrate_fixedk.py and heldout_hooks.py.
"""
import os
import urllib.request

import yaml

COMMON = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.dirname(COMMON)
DATA = os.environ.get("CODEEVOLVE_DATA", os.path.expanduser("~/codeEvolve/data"))
COLOR02 = "https://mat.tepper.cmu.edu/COLOR02/INSTANCES/"

# best-known / optimal colors (Goudet & Hao 2021, Titiloye QA-COL certificates, COLOR02 table)
BEST_KNOWN = {
    "myciel3": 4, "myciel4": 5, "myciel5": 6, "myciel6": 7, "myciel7": 8,
    "queen5_5": 5, "queen6_6": 7, "queen7_7": 7, "queen8_8": 9, "queen9_9": 10, "queen10_10": 11,
    "DSJC125.1": 5, "DSJC125.5": 17, "DSJC125.9": 44,
    "DSJC250.1": 8, "DSJC250.5": 28, "DSJC250.9": 72,
    "DSJC500.1": 12, "DSJC500.5": 47, "DSJC500.9": 126,
    "DSJC1000.1": 20, "DSJC1000.5": 82, "DSJC1000.9": 222,
    "DSJR500.1c": 85, "DSJR500.5": 122,
    "le450_15c": 15, "le450_15d": 15, "le450_25c": 25, "le450_25d": 25,
    "flat300_28_0": 28, "flat1000_50_0": 50, "flat1000_60_0": 60, "flat1000_76_0": 81,
    "latin_square_10": 97, "C2000.5": 145, "C4000.5": 259,
}


def lm(model, temp):
    return {"model_name": model, "temp": temp, "top_p": 0.8, "max_tok": 8192, "retries": 3, "weight": 1, "verify_ssl": False}


class BlockStr(str):
    pass


def _block_repr(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(BlockStr, _block_repr)


def write_yaml(path, cfg):
    cfg = dict(cfg)
    cfg["SYS_MSG"] = BlockStr(cfg["SYS_MSG"])
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, width=120)


def ensure_graph(instance):
    os.makedirs(os.path.join(DATA, "dimacs"), exist_ok=True)
    path = os.path.join(DATA, "dimacs", f"{instance}.col")
    if not os.path.exists(path):
        url = f"{COLOR02}{instance}.col"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path
