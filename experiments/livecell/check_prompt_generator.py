import vigra
import napari
import imageio

from micro_sam.util import get_cell_center_coordinates
from micro_sam.prompt_generators import PointPromptGenerator


def check_prompt_generator(image_path, label_path, view):
    image = imageio.imread(image_path)
    segmentation = imageio.imread(label_path)
    segmentation, _, _ = vigra.analysis.relabelConsecutive(segmentation.astype("uint32"))

    seg_id = 49
    mask = segmentation == seg_id + 1

    prompt_generator = PointPromptGenerator(n_positive_points=1, n_negative_points=0, dilation_strength=3)

    centers, boxes = get_cell_center_coordinates(segmentation)

    center, box = centers[seg_id], boxes[seg_id]

    points, labels, _ = prompt_generator(segmentation, seg_id, center, box)

    if view:
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        v.add_labels(mask)
        v.add_points(points)
        napari.run()


def main():
    image_path = "C:/MICCAI 2023/images/images/A172_Phase_C7_1_00d00h00m_1.tif"
    label_path = "C:/MICCAI 2023/images/annotations/livecell_test_images/A172/A172_Phase_C7_1_00d00h00m_1.tif"

    check_prompt_generator(image_path, label_path, view=False)


if __name__ == "__main__":
    main()
