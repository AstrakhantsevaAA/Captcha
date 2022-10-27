import random
from pathlib import Path

import cv2
import numpy as np
import typer


def noisy(noise_type: int, image: np.ndarray) -> np.ndarray:
    if noise_type == "gauss" or noise_type == 1:
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape((row, col, ch))
        out = image + gauss
        return out
    elif noise_type == "s&p" or noise_type == 2:
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:-1]]
        out[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [
            np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:-1]
        ]
        out[tuple(coords)] = 0
        return out
    elif noise_type == "poisson" or noise_type == 3:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        out = np.random.poisson(image * vals) / float(vals)
        return out
    elif noise_type == "speckle" or noise_type == 4:
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape((row, col, ch))
        out = image + image * gauss
        return out
    else:
        raise Exception(
            f"Expected noise_type: 1 or 'gauss', 2 or 's&p', 3 or 'poisson', 4 or 'speckle', got {noise_type}"
        )


def generate_one_image():
    shape = (50, 200, 1)
    image = np.zeros(shape)
    font = random.randint(0, 7)
    org = (30, 35)
    font_scale = 1
    color = random.randint(200, 255)
    thickness = random.randint(0, 4)

    text = "".join([str(random.randint(0, 9)) for _ in range(6)])

    image = cv2.putText(
        image, text, org, font, font_scale, color, thickness, cv2.LINE_AA
    )
    # invert to get image as test images
    image = 255 - image
    type_noise = random.randint(1, 4)
    image = noisy(type_noise, image)
    # pixel values should be in [0, 255]
    image = np.where(image > 255, 255, image)
    image = np.where(image < 0, 0, image)

    return image, text


def generate_data(save_path: str, number_images: int):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    for _ in range(number_images):
        img, label = generate_one_image()
        cv2.imwrite(f"{save_path}/{label}.png", img)


if __name__ == "__main__":
    typer.run(generate_data)
