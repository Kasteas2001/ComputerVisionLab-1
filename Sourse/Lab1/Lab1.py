import cv2
import numpy as np
from PIL import Image, ImageEnhance

def apply_rotation(img, deg):
    (h, w) = img.shape[:2]
    c = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(c, deg, 1.0)
    return cv2.warpAffine(img, m, (w, h))

def flip_image(img, direction):
    directions = {'horizontal': 1, 'vertical': 0}
    return cv2.flip(img, directions.get(direction, 1))

def resize_image(img, scale):
    (h, w) = img.shape[:2]
    dims = (int(w * scale), int(h * scale))
    return cv2.resize(img, dims, interpolation=cv2.INTER_LINEAR)

def central_crop(img):
    (h, w) = img.shape[:2]
    return img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

def translate_image(img, dx, dy):
    (h, w) = img.shape[:2]
    m_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m_shift, (w, h))

def gaussian_blur(img, k_size):
    return cv2.GaussianBlur(img, (k_size, k_size), 0)

def to_black_white(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    return bw_img

def make_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def add_tint(img, color):
    channel_map = {'red': 2, 'green': 1, 'blue': 0}
    tinted_img = np.zeros_like(img)
    tinted_img[:, :, channel_map[color]] = img[:, :, channel_map[color]]
    return tinted_img

def sepia_filter(img):
    k = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, k)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def change_saturation(img, level):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    color_adjuster = ImageEnhance.Color(img_pil)
    new_img = color_adjuster.enhance(level)
    return cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)

def reverse_colors(img):
    return cv2.bitwise_not(img)

def hsv_modification(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    v = np.clip(v + 50, 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def images_to_pdf(images, pdf_file):
    pil_imgs = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]
    pil_imgs[0].save(pdf_file, save_all=True, append_images=pil_imgs[1:])

if __name__ == "__main__":
    images_processed = []
    img_original = cv2.imread('input.webp')

    operations = [
        (apply_rotation, 90),
        (apply_rotation, -90),
        (flip_image, 'horizontal'),
        (flip_image, 'vertical'),
        (resize_image, 2),
        (resize_image, 0.5),
        (central_crop, ),
        (apply_rotation, -30),
        (translate_image, 50, 25),
        (gaussian_blur, 15),
        (to_black_white, ),
        (make_grayscale, ),
        (add_tint, 'red'),
        (add_tint, 'green'),
        (add_tint, 'blue'),
        (sepia_filter, ),
        (change_saturation, -0.5),
        (change_saturation, 1.5),
        (reverse_colors, ),
        (hsv_modification, )
    ]

    for func, *args in operations:
        images_processed.append(func(img_original, *args))

    images_to_pdf(images_processed, 'output.pdf')
    print("PDF создан: output.pdf")
