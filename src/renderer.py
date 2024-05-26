import mitsuba as mi
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
mi.set_variant("cuda_ad_rgb")


def save_images_with_varying_spp(scene, scene_name, num_steps=1000):
    images = []
    gt_image = mi.render(scene, spp=4096)
    for spp in range(1, 1024, 1024 // num_steps):
        image = mi.render(scene, spp=spp)
        mi.util.write_bitmap(
            f"../data/cbox/train_{spp-1}.png", image, write_async=True)

    return images


if __name__ == "__main__":
    scene = mi.load_file("../mitsuba_data/simple_examples/cbox.xml")

    save_images_with_varying_spp(scene, "cbox", num_steps=1000)

