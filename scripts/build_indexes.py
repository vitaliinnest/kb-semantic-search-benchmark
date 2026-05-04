"""
Build FAISS indexes for all embedding models across all 3 domains.
Runs sequentially to avoid memory issues on CPU-only machine.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BUILD_SCRIPT = ROOT / "src" / "build_index.py"

DOMAINS = ["tech", "legal", "medical"]

# (artifact_folder, model_type, model_name, batch_size, max_seq_length)
MODELS = [
    ("e5-base",  "e5",    "intfloat/multilingual-e5-base",   32, None),
    ("nomic",    "nomic", "nomic-ai/nomic-embed-text-v1.5",   32, 512),
    ("bge-m3",   "sbert", "BAAI/bge-m3",                      8,  256),
    ("qwen3",    "sbert", "Qwen/Qwen3-Embedding-0.6B",         4,  256),
]

def run_build(domain, artifact_id, model_type, model_name, batch_size, max_seq_length):
    chunks = ROOT / "data" / "domains" / domain / "chunks.jsonl"
    artifacts = ROOT / "artifacts" / domain / artifact_id

    if not chunks.exists():
        print(f"  SKIP: chunks not found at {chunks}")
        return False

    # Check if already built
    if (artifacts / "faiss.index").exists():
        print(f"  SKIP: already built at {artifacts}")
        return True

    artifacts.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PYTHON), str(BUILD_SCRIPT),
        "--model-type", model_type,
        "--model", model_name,
        "--chunks", str(chunks),
        "--artifacts", str(artifacts),
        "--batch-size", str(batch_size),
    ]
    if max_seq_length:
        cmd += ["--max-seq-length", str(max_seq_length)]

    print(f"  CMD: {' '.join(cmd[-6:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  OK  ({elapsed/60:.1f} min)")
        return True
    else:
        print(f"  FAIL (exit={result.returncode}) — {elapsed/60:.1f} min")
        last_err = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        for line in last_err:
            print(f"    {line}")
        return False


def main():
    print("=" * 60)
    print("Building model indexes")
    print("=" * 60)
    results = {}

    for artifact_id, model_type, model_name, batch_size, max_seq_length in MODELS:
        print(f"\n[MODEL] {artifact_id}  ({model_name})")
        results[artifact_id] = {}
        for domain in DOMAINS:
            print(f"  [DOMAIN] {domain}")
            ok = run_build(domain, artifact_id, model_type, model_name, batch_size, max_seq_length)
            results[artifact_id][domain] = "OK" if ok else "FAIL"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, domres in results.items():
        for domain, status in domres.items():
            print(f"  {model:12} / {domain:8}: {status}")


if __name__ == "__main__":
    main()
