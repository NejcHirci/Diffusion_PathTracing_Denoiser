import mitsuba as mi
import tqdm

#matplotlib.use("TkAgg")
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


def save_images_with_varying_spp_and_position(scene, scene_name, shift_or_rot="shift", num_seeds=4, num_angles=8):
    spp_list = [1, 2, 4, 8, 16, 32, 64, 128]
    params = mi.traverse(scene)

    print(f"Saving images for {scene_name}")
    with tqdm.tqdm(total=num_seeds*num_angles*len(spp_list)) as pbar:
            for t in range(num_angles):
                transform = mi.Transform4f.rotate(axis=[0, 1, 0], angle=-30/num_angles) if shift_or_rot == "rot" else mi.Transform4f.translate([0, 0, 5/num_angles])
                params["sensor.to_world"] @= transform
                params.update()
                gt_image = mi.render(scene, spp=4096, seed=0)
                mi.util.write_bitmap(f"../data/{scene_name}/gt_{t}.png", gt_image, write_async=False)

                for seed in range(num_seeds):
                    for spp in spp_list:
                        image = mi.render(scene, spp=spp, seed=seed)
                        mi.util.write_bitmap(
                            f"../data/{scene_name}/train_{seed}_{t}_{spp}.png", image, write_async=False)
                        pbar.update(1)
    print(f"Done with {scene_name}")



if __name__ == "__main__":
    cbox_scene = mi.load_file("../mitsuba_data/simple_examples/scene.xml")
    bathroom_scene = mi.load_file("../mitsuba_data/bathroom/scene.xml")
    kitchen_scene = mi.load_file("../mitsuba_data/kitchen/scene.xml")
    veach_ajar_scene = mi.load_file("../mitsuba_data/veach_ajar/scene.xml")
    veach_bidir_scene = mi.load_file("../mitsuba_data/veach-bidir/scene.xml")
    house_scene = mi.load_file("../mitsuba_data/house/scene.xml")
    living_room_scene = mi.load_file("../mitsuba_data/living-room-2/scene.xml")

    #scenes = [cbox_scene,bathroom_scene, kitchen_scene, veach_ajar_scene,
              #veach_bidir_scene]
    #scene_names = ["cbox_novel","bathroom_novel", "kitchen_novel", "veach_ajar_novel", "veach_bidir_novel"]
    #shift_or_rot = ["shift", "rot", "shift", "shift", "shift"]

    scene_names = ["veach_bidir_novel", "house_novel", "living_room_novel"]
    scenes = [veach_bidir_scene, house_scene, living_room_scene]
    shift_or_rot = ["shift", "shift", "rot"]

    # Veach bidir shift 5
    # House shift 30
    # living room shift 5

    save_images_with_varying_spp_and_position(living_room_scene, "living_room_novel", "shift", num_seeds=8, num_angles=8)

