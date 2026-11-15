"""
Execute notebooks and save with embedded outputs.
"""
import sys, os, glob, time
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks")

notebook_files = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))

for nb_path in notebook_files:
    fname = os.path.basename(nb_path)
    print(f"\n==================================================", flush=True)
    print(f"Executing: {fname}", flush=True)
    print(f"==================================================", flush=True)
    t0 = time.time()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="pestalertnet",
        resources={"metadata": {"path": PROJECT_ROOT}},
    )
    
    try:
        client.execute()
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"✓ Completed {fname} in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"✗ Failed {fname} after {time.time()-t0:.1f}s: {e}", flush=True)
        # Re-raise to see full trace
        raise

print("\n🎉 All notebooks executed successfully with embedded outputs!")
