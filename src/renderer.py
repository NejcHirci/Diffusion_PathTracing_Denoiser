import mitsuba as mi
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
mi.set_variant("cuda_ad_rgb")


def save_images_with_varying_spp(scene, scene_name, num_seeds=8):
    images = []
    gt_image = mi.render(scene, spp=4096, seed=0)
    mi.util.write_bitmap(f"../data/{scene_name}/gt.png", gt_image, write_async=False)
    for i in range(num_seeds):
        spp_list = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 56, 64, 96, 128, 160]
        for spp in spp_list:
            image = mi.render(scene, spp=spp, seed=i+1)
            mi.util.write_bitmap(
                f"../data/{scene_name}/train_{i+1}_{spp}.png", image, write_async=False)

    return images


if __name__ == "__main__":
    cbox_scene = mi.load_file("../mitsuba_data/simple_examples/scene.xml")
    bathroom_scene = mi.load_file("../mitsuba_data/bathroom/scene.xml")
    kitchen_scene = mi.load_file("../mitsuba_data/kitchen/scene.xml")
    veach_ajar_scene = mi.load_file("../mitsuba_data/veach_ajar/scene.xml")
    veach_bidir_scene = mi.load_file("../mitsuba_data/veach-bidir/scene.xml")

    scenes = [cbox_scene]#, bathroom_scene, kitchen_scene, veach_ajar_scene,
              #veach_bidir_scene]
    scene_names = ["cbox", "bathroom", "kitchen", "veach_ajar", "veach_bidir"]
    for i in range(len(scenes)):
        save_images_with_varying_spp(scenes[i], scene_names[i])

