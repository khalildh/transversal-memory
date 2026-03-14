"""Quick sweep: decay values (fast mode) + longer sequences."""
import subprocess, sys

runs = [
    # Decay sweep (fast 2-layer model, ~2 min each)
    ("decay=0.95", ["--fast", "--from-scratch", "--decay", "0.95", "--epochs", "7"]),
    ("decay=0.99", ["--fast", "--from-scratch", "--decay", "0.99", "--epochs", "7"]),
    ("decay=0.995", ["--fast", "--from-scratch", "--decay", "0.995", "--epochs", "7"]),
    ("decay=1.0", ["--fast", "--from-scratch", "--decay", "1.0", "--epochs", "7"]),
    # Standard baseline (fast, for comparison)
    ("standard", ["--fast", "--from-scratch", "--epochs", "7"]),
]

print("=" * 60, flush=True)
print("  SWEEP: decay tuning + standard baseline (fast mode)", flush=True)
print("=" * 60, flush=True)

results = {}
for name, args in runs:
    print(f"\n--- {name} ---", flush=True)
    variant = "standard" if name == "standard" else "online_mem"
    cmd = [sys.executable, "-u", "exp_fast.py", variant] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    print(output, flush=True)
    # Extract best PPL
    for line in output.split("\n"):
        if "best PPL" in line:
            ppl = float(line.split("=")[1].strip())
            results[name] = ppl

print("\n" + "=" * 60, flush=True)
print("  RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
for name, ppl in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:15s}: PPL {ppl:.1f}", flush=True)
