"""
Train Jarvis personality LoRA adapters on local Qwen3-1.7B.
Uses mlx_lm.lora — runs entirely on Apple Silicon Metal, no GPU needed.
Produces one LoRA adapter per personality (5 total).

Each adapter IS a personality — not injected text, actual learned weights.

Run:
    python3.14 train_lora.py                         # trains all 5 (~2-4h total)
    python3.14 train_lora.py --personality explorer  # trains one (~30-45 min)
    python3.14 train_lora.py --status                # check which are done
    python3.14 train_lora.py --test explorer         # test a trained adapter
    python3.14 train_lora.py --fuse explorer         # fuse adapter into base model
"""

import subprocess
import sys
import time
import yaml
from pathlib import Path

_ROOT        = Path(__file__).parent
DATA_DIR     = _ROOT / "data"
ADAPTERS_DIR = _ROOT / "lora_adapters"
FUSED_DIR    = _ROOT / "fused_models"
ADAPTERS_DIR.mkdir(exist_ok=True)
FUSED_DIR.mkdir(exist_ok=True)

MODEL_PATH = (
    "/Volumes/ExternalDisk/huggingface/hub"
    "/models--Qwen--Qwen3-1.7B/snapshots"
    "/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)

PERSONALITIES = ["explorer", "scientist", "critic", "synthesiser", "pragmatist"]

RANK  = 16
ITERS = 300


def print_banner():
    print(f"\n{'━'*60}")
    print(f"  JARVIS PERSONALITY LORA TRAINING")
    print(f"  Model: Qwen3-1.7B  |  Rank: {RANK}  |  Iters: {ITERS}")
    print(f"  Personalities: {', '.join(PERSONALITIES)}")
    print(f"{'━'*60}\n")


def write_config(personality: str, data_dir: Path, adapter_dir: Path) -> Path:
    cfg = {
        "model":            MODEL_PATH,
        "train":            True,
        "data":             str(data_dir),
        "adapter_path":     str(adapter_dir),
        "num_layers":       RANK,
        "batch_size":       4,
        "iters":            ITERS,
        "learning_rate":    2e-5,
        "val_batches":      10,
        "save_every":       100,
        "steps_per_report": 10,
        "grad_checkpoint":  True,
        "seed":             42,
        "lora_parameters": {
            "rank":    RANK,
            "dropout": 0.05,
            "scale":   20.0,
        },
    }
    cfg_path = _ROOT / f"config_{personality}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path


def prepare_data_dir(personality: str) -> Path:
    src = DATA_DIR / "personality_train.jsonl"
    if not src.exists():
        print("  No training data. Generating...")
        subprocess.run([sys.executable, str(_ROOT / "generate_training_data.py")])

    p_data_dir = DATA_DIR / personality
    p_data_dir.mkdir(exist_ok=True)

    all_lines = open(src).readlines()
    p_idx  = PERSONALITIES.index(personality)
    p_lines = [l for i, l in enumerate(all_lines) if i % len(PERSONALITIES) == p_idx]

    if not p_lines:
        p_lines = all_lines

    split = max(1, len(p_lines) * 4 // 5)
    train = p_lines[:split]
    valid = p_lines[split:] or p_lines[:1]

    (p_data_dir / "train.jsonl").write_text("".join(train))
    (p_data_dir / "valid.jsonl").write_text("".join(valid))

    print(f"  {personality}: {len(train)} train, {len(valid)} valid examples")
    return p_data_dir


def train_personality(personality: str) -> Path:
    print(f"\n{'─'*60}")
    print(f"  Training: {personality}  |  model: Qwen3-1.7B  |  iters: {ITERS}")
    print(f"{'─'*60}")

    data_dir    = prepare_data_dir(personality)
    adapter_dir = ADAPTERS_DIR / personality
    adapter_dir.mkdir(exist_ok=True)
    cfg_path    = write_config(personality, data_dir, adapter_dir)

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfg_path)]
    print(f"  Command: {' '.join(cmd)}\n")
    t0 = time.time()

    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✓ {personality} done in {elapsed/60:.1f} min → {adapter_dir}")
    else:
        print(f"\n  ✗ {personality} failed (exit {result.returncode})")
        print(f"  Config left at: {cfg_path}")
        return adapter_dir

    cfg_path.unlink(missing_ok=True)
    return adapter_dir


def fuse_adapter(personality: str):
    """Merge LoRA weights into base model for faster inference."""
    adapter_dir = ADAPTERS_DIR / personality
    if not (adapter_dir / "adapters.safetensors").exists():
        print(f"  No adapter for {personality}. Train first.")
        return

    out_dir = FUSED_DIR / personality
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model",        MODEL_PATH,
        "--adapter-path", str(adapter_dir),
        "--save-path",    str(out_dir),
    ]
    print(f"  Fusing {personality} → {out_dir}")
    subprocess.run(cmd)


def test_adapter(personality: str):
    adapter_dir = ADAPTERS_DIR / personality
    fused_dir   = FUSED_DIR / personality

    if (fused_dir / "config.json").exists():
        model_arg = str(fused_dir)
        extra     = []
        print(f"\n  Testing [{personality}] (fused model)\n")
    elif (adapter_dir / "adapters.safetensors").exists():
        model_arg = MODEL_PATH
        extra     = ["--adapter-path", str(adapter_dir)]
        print(f"\n  Testing [{personality}] (adapter)\n")
    else:
        print(f"  No adapter found for {personality}. Train first.")
        return

    prompt = "Why can axolotls regenerate limbs but humans cannot?"
    print(f"  Prompt: '{prompt}'\n")

    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model",      model_arg,
        "--prompt",     prompt,
        "--max-tokens", "250",
        "--temp",       "0.7",
    ] + extra
    subprocess.run(cmd)


def status():
    print("  LoRA Adapter Status:")
    print("  " + "─" * 48)
    any_done = False
    for p in PERSONALITIES:
        adapter_dir = ADAPTERS_DIR / p
        fused_dir   = FUSED_DIR / p
        has_adapter = (adapter_dir / "adapters.safetensors").exists()
        has_fused   = (fused_dir / "config.json").exists()
        any_done    = any_done or has_adapter or has_fused
        a_str = "✓" if has_adapter else "✗"
        f_str = "✓" if has_fused   else "✗"
        print(f"  {p:<15} adapter={a_str}  fused={f_str}")

    if not any_done:
        print("\n  No adapters trained yet.")
        print("  Run:  python3.14 train_lora.py --personality explorer")
        print("  Full: python3.14 train_lora.py")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        print_banner()
        status()
        sys.exit(0)

    if "--fuse" in args:
        idx = args.index("--fuse")
        p   = args[idx + 1] if idx + 1 < len(args) else "explorer"
        fuse_adapter(p)
        sys.exit(0)

    if "--test" in args:
        idx = args.index("--test")
        p   = args[idx + 1] if idx + 1 < len(args) else "explorer"
        test_adapter(p)
        sys.exit(0)

    if "--personality" in args:
        idx  = args.index("--personality")
        todo = [args[idx + 1]]
    else:
        todo = PERSONALITIES

    print_banner()
    status()
    print()

    for p in todo:
        train_personality(p)

    print(f"\n{'━'*60}")
    print("  Done. Next steps:")
    for p in todo:
        print(f"    python3.14 train_lora.py --test  {p}")
        print(f"    python3.14 train_lora.py --fuse  {p}")
    print(f"\n  Run all personalities:")
    print(f"    python3.14 lora_inference.py")
    print(f"{'━'*60}")
