import argparse
import json
from src import Baseline, PoseEstimator, evaluate, print_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the YCB-V test targets.")
    parser.add_argument("checkpoint")
    parser.add_argument("--variant", choices=["baseline", "rgbd"], default="rgbd")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=1024, help="model points sampled per object")
    parser.add_argument("--json", help="also write the full results to this path")
    args = parser.parse_args()

    model = Baseline() if args.variant == "baseline" else PoseEstimator()
    results = evaluate(model, args.checkpoint, args.variant, bs=args.bs, num_points=args.num_points)
    print_results(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
