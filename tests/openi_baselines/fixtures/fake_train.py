import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--pred-len", type=int, required=True)
parser.add_argument("--fail", action="store_true")
args = parser.parse_args()

Path("checkpoint.pth").write_bytes(b"weights")
Path("pred.npy").write_bytes(b"prediction")

if args.fail:
    print(f"intentional failure for pred_len={args.pred_len}", flush=True)
    raise SystemExit(7)

mse = args.pred_len / 1000
mae = args.pred_len / 2000
line = f"mse:{mse}, mae:{mae}"
Path("result.txt").write_text(line + "\n", encoding="utf-8")
print(line, flush=True)
