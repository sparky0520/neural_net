import argparse
import train_mnist
import test_mnist


def main():
    parser = argparse.ArgumentParser(description="Neural Network MNIST CLI")
    parser.add_argument(
        "--test", action="store_true", help="Only run the test set on the saved model"
    )
    args = parser.parse_args()

    if args.test:
        test_mnist.test_saved_model()
    else:
        print("Starting full pipeline (Train + Test)...")
        train_mnist.train()
        test_mnist.test_saved_model()


if __name__ == "__main__":
    main()
