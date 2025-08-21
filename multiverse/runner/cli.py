import argparse
import os
from .docker_runner import run_model_container

def main():
    parser = argparse.ArgumentParser(description="Run multiverse models in Docker containers")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to run (e.g., pca mofa multivi)")
    parser.add_argument("--input", required=True, help="Path to the input data directory")
    parser.add_argument("--output", required=True, help="Path to the output results directory")
    args = parser.parse_args()

    print(f"Running models: {args.models}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")

    # Create the base output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    for model in args.models:
        print(f"\n=== Running model: {model} ===")
        try:
            run_model_container(model, args.input, args.output)
            print(f"=== Model {model} finished successfully ===")
        except Exception as e:
            print(f"!!! Error running model {model}: {e} !!!")
            # Decide if you want to continue with other models or stop
            # For now, we'll just print the error and continue
            pass

if __name__ == "__main__":
    main()
