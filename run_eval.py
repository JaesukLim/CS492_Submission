## This code is from https://github.com/KAIST-Visual-AI-Group/Diffusion-Project-Drawing/blob/master/run_eval.py

import argparse
import json
from cleanfid import fid

parser = argparse.ArgumentParser()
parser.add_argument("--fdir1", type=str, default="./test_data")
parser.add_argument("--fdir2", type=str, default="./test_data_2")
parser.add_argument("--save_dir", type=str, default="./result")
args = parser.parse_args()

# compute FID
score_fid = fid.compute_fid(args.fdir1, args.fdir2)

# compute KID
score_kid = fid.compute_kid(args.fdir1, args.fdir2)

print("========================")
print(f"- FID score: {score_fid}")
print(f"- KID score: {score_kid}")
print("========================")

result = {
    "fdir1": args.fdir1,
    "fdir2": args.fdir2,
    "FID": score_fid,
    "KID": score_kid
}

with open(f"{args.save_dir}/result.json", "w") as f:
    json.dump(result, f, indent=2)