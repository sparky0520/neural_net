import numpy as np
from PIL import Image, ImageOps
from network import NeuralNetwork
import os


def predict_image(image_path, model_path="mnist_model.json"):
    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Please train the model first."
        )
        return

    # 1. Load the model
    net = NeuralNetwork.load(model_path)

    # 2. Preprocess the image
    try:
        # Open image and convert to grayscale ('L' mode)
        img = Image.open(image_path).convert("L")

        # Resize to 28x28 using LANCZOS for high quality
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Most users draw black ink on white paper.
        # MNIST is white ink on black background.
        # We check the corner pixel to decide if we need to invert.
        if img.getpixel((0, 0)) > 128:
            img = ImageOps.invert(img)

        # Convert to numpy array and normalize [0, 1]
        data = np.array(img).reshape(784, 1) / 255.0

        # 3. Predict
        prediction = net.prediction(data)
        digit = np.argmax(prediction)
        confidence = prediction[digit][0] * 100

        print("\n--- Prediction Results ---")
        print(f"Predicted Digit: {digit}")
        print(f"Confidence: {confidence:.2f}%")

        # Show mini visualization in terminal
        # (Very helpful for debugging)
        print("\nWhat the network saw (28x28 thumbnail):")
        for row in range(28):
            line = ""
            for col in range(28):
                val = data[row * 28 + col][0]
                if val > 0.75:
                    line += "##"
                elif val > 0.5:
                    line += "++"
                elif val > 0.25:
                    line += ".."
                else:
                    line += "  "
            print(line)

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
