
import glob

from PIL import Image


def load_image_1():
    image_path = glob.glob('../assets/images/*.jpg')
    mask_path = glob.glob('../assets/masks/*.jpg')

    image_open = Image.open(image_path[0])
    mask_open = Image.open(mask_path[0])

    image = image_open.resize((352, 352))
    image_LANCZOS = image_open.resize((352, 352), resample=Image.Resampling.LANCZOS)

    mask = mask_open.resize((352, 352))
    mask_LANCZOS = mask_open.resize((352, 352), resample=Image.Resampling.LANCZOS)

    image.save('1-image_loader/image.png')
    image_LANCZOS.save('1-image_loader/image_LANCZOS.png')
    mask.save('1-image_loader/mask.png')
    mask_LANCZOS.save('1-image_loader/mask_LANCZOS.png')


def load_image_2():
    cat_open = Image.open('img.png')
    flower_open = Image.open('img_1.png')

    cat = cat_open.resize((352, 352))
    cat_LANCZOS = cat_open.resize((352, 352), resample=Image.Resampling.LANCZOS)

    flower = flower_open.resize((352, 352))
    flower_LANCZOS = flower_open.resize((352, 352), resample=Image.Resampling.LANCZOS)

    cat.show()
    cat_LANCZOS.show()
    flower.show()
    flower_LANCZOS.show()

    cat.save('1-image_loader/cat.png')
    cat_LANCZOS.save('1-image_loader/cat_LANCZOS.png')
    flower.save('1-image_loader/flower.png')
    flower_LANCZOS.save('1-image_loader/flower_LANCZOS.png')


# load_image_1()
# load_image_2()

