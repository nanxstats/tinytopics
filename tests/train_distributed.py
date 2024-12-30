import argparse
from pathlib import Path

import torch
from accelerate.utils import set_seed

from tinytopics.fit_distributed import fit_model_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_topics", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Load data
    X = torch.load(args.data_path)

    # Run training
    model, losses = fit_model_distributed(
        X=X,
        k=args.num_topics,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )

    # Save losses for verification
    if args.save_path:
        save_dir = Path(args.save_path).parent
        losses_path = (
            save_dir / f"losses{Path(args.save_path).stem.replace('model', '')}.pt"
        )
        torch.save(losses, losses_path)

    print("Training completed successfully")


if __name__ == "__main__":
    main()
