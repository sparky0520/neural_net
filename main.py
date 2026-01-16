import argparse
import train_mnist
import test_mnist
import predict


def main():
    parser = argparse.ArgumentParser(description="Neural Network MNIST CLI")
    parser.add_argument(
        "--test", action="store_true", help="Only run the test set on the saved model"
    )
    parser.add_argument(
        "--predict", type=str, help="Path to an image file to predict its digit"
    )
    args = parser.parse_args()

    if args.predict:
        predict.predict_image(args.predict)
    elif args.test:
        test_mnist.test_saved_model()
    else:
        print("Starting full pipeline (Train + Test)...")
        train_mnist.train()
        test_mnist.test_saved_model()


if __name__ == "__main__":
    main()
