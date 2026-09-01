import torch
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("path", type=str, help="Path to checkpoint")
args = ap.parse_args()

c = torch.load(args.path)

count = 0

def get_count(o):
    if isinstance(o, torch.Tensor):
        return o.numel()
    elif isinstance(o, dict):
        ans = 0
        for k in o:
            ans += get_count(o[k])
        return ans
    else:
        return 0

if isinstance(c, dict):
    for k in c:
        pc = get_count(c[k])
        print(f"Module {k} has {pc} parameters")
        count += pc
else:
    count = get_count(c)

print(f"Total number of parameters: {count}")