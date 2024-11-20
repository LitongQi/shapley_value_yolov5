import os
import os.path as osp
import shutil
import numpy as np
import cv2
from tqdm import tqdm


from detect import run as run_detect
from shapley_main import main as shapley_main
from shapley_functions import monte_carlo_shapley_value, kernel_shapley_additive_value, shapley_additive_value


default_kwargs = {
    "weights": "yolov5s.pt",
    "imgsz": (640, 640),
    "conf_thres": 0.25,
    "save_csv": True, 
    "save_txt": True,
    "save_format": 1,
}

shapley_function_dict = {
    "monte-carlo": monte_carlo_shapley_value,
    "additive": shapley_additive_value,
    "kernel": kernel_shapley_additive_value
}


def random_mask(img, patch_size, mask_ratio):
    h, w = img.shape[:2]
    assert h % patch_size == 0 and w % patch_size == 0
    h0, w0 = h//patch_size, w//patch_size

    mask = (np.random.rand(h0, 1, w0, 1, 1) > mask_ratio).astype(float)
    img = img.reshape(h0, patch_size, w // patch_size, patch_size, -1)
    masked_img = img * mask + 127 * (1 - mask)
    masked_img = masked_img.reshape((h,w,-1))

    return masked_img, mask.reshape((h0, w0))

def generate_data(source_img="./shap/bus.jpg", patch_size=32, num_samples=10000, output_dir="./shap/exp"):
    keep_features = []
    img = cv2.imread(source_img)
    h_ori, w_ori = img.shape[:2]
    h, w = h_ori // patch_size * patch_size, w_ori // patch_size * patch_size
    img = cv2.resize(img, (w, h))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, "original_image.jpg"), img)

    for i in tqdm(range(num_samples)):
        mask_ratio = np.random.rand() * 0.3
        masked_img, mask = random_mask(img, patch_size, mask_ratio)
        keep_features.append(mask)
        cv2.imwrite(os.path.join(output_dir, "%06d.jpg"%i), masked_img)

    np.save(output_dir + "/masks.npy", np.array(keep_features))

def vis_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    norm_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    norm_heatmap = norm_heatmap.astype(np.uint8)
    heat_img = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

    vis_img = cv2.addWeighted(image, 0.3, heat_img, 0.7, 0)

    return vis_img

def generate_data_and_detect(source_img, mask_output_dir, detect_dir, num_samples):
    # generate masked images
    if osp.exists(mask_output_dir):
        shutil.rmtree(mask_output_dir)
    print("\nGenerating masked images (%d images in total)..."%num_samples)
    generate_data(source_img=source_img, num_samples=num_samples, output_dir=mask_output_dir)

    # detect objects in each masked image
    if osp.exists(detect_dir):
        shutil.rmtree(detect_dir)
    default_kwargs["source"] = mask_output_dir
    default_kwargs["project"] = detect_dir
    print("\nDetecting objects in masked images...")
    run_detect(**default_kwargs)


def main():
    # config
    image_name = "bus.jpg"
    source_img = "./shap/%s"%image_name
    mask_output_dir = "./shap/%s/masked_images"%image_name.split(".")[0]
    detect_dir = "./shap/%s/detect_results"%image_name.split(".")[0]
    shap_vis_output_dir = "./shap/%s/shapley_value_visualizations"%image_name.split(".")[0]

    per_object = True
    num_samples = 10000
    iou_thresholds = [0.3, 0.5, 0.7]
    shapley_functions = ["monte-carlo", "additive", "kernel"]


    # generate masked images and detect the objects
    generate_data_and_detect(source_img, mask_output_dir, detect_dir, num_samples)


    # compute the shapley values
    if osp.exists(shap_vis_output_dir):
        shutil.rmtree(shap_vis_output_dir)
    os.makedirs(shap_vis_output_dir)

    ori_img = cv2.imread(osp.join(mask_output_dir, "original_image.jpg"))
    for shapley_function in shapley_functions:
        for iou_thres in iou_thresholds:
            print("\nRunning %s with iou threshold %.2f"%(shapley_function, iou_thres))
            print("It might takes a few minutes...")
            shap_values = shapley_main(
                detect_dir, 
                os.path.join(mask_output_dir, "masks.npy"), 
                shapley_function_dict[shapley_function], 
                per_object, 
                iou_thres
            )

            # visualization
            for i in range(len(shap_values)):
                vis_img = vis_heatmap(ori_img, shap_values[i])
                cv2.imwrite(osp.join(shap_vis_output_dir, "shap_%s_thres_%.2f_obj_%03d.jpg"%(shapley_function, iou_thres, i)), vis_img)
            print("Done.")

if __name__ == '__main__':
    main()
