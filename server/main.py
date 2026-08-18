"""Root inference/demo entry → scripts/infer.py (load checkpoint, encode scene, predict, optional Unity).

  python main.py --ckpt outputs/final/best.pt --scene hub:fire_blocked:14:123
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "infer.py"),
               run_name="__main__")
