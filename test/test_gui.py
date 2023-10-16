# import numpy as np
# import skimage.data
# from micro_sam.sam_annotator import annotator_2d, annotator_3d
# from micro_sam.sam_annotator.annotator_2d import _initialize_viewer, _segment_widget, _autosegment_widget
# from micro_sam.sam_annotator.util import _clear_widget, _commit_segmentation_widget


# def _check_layer_initialization(viewer):
#     """Utility function to check the initial layer setup is correct."""
#     assert len(viewer.layers) == 6
#     expected_layer_names = ['raw', 'auto_segmentation', 'committed_objects', 'current_object', 'point_prompts', 'prompts']
#     for layername in expected_layer_names:
#         assert layername in viewer.layers
#     # Check layers are empty before beginning tests
#     np.testing.assert_equal(viewer.layers["auto_segmentation"].data, 0)
#     np.testing.assert_equal(viewer.layers["current_object"].data, 0)
#     np.testing.assert_equal(viewer.layers["committed_objects"].data, 0)
#     np.testing.assert_equal(viewer.layers["point_prompts"].data, 0)
#     assert viewer.layers["prompts"].data == []  # shape data is list, not numpy array


# def test_annotator_2d_amg(make_napari_viewer_proxy, tmp_path):
#     """Integration test for annotator_2d widget with automatic mask generation.

#     * Creates 2D image embedding
#     * Opens annotator_2d widget in napari
#     * Test automatic mask generation
#     """
#     model_type = "vit_b"
#     embedding_path = tmp_path / "test-embedding.zarr"
#     # example data - a basic checkerboard pattern
#     image = np.zeros((16,16,16))
#     image[:8,:8,:8] = 1
#     image[8:,8:,8:] = 1

#     viewer = make_napari_viewer_proxy()
#     viewer = _initialize_viewer(image, None, None, None)  # TODO: fix hacky workaround
#     # test generating image embedding, then adding micro-sam dock widgets to the GUI
#     viewer = annotator_2d(
#         image,
#         embedding_path,
#         show_embeddings=False,
#         model_type=model_type,
#         v=viewer,
#         return_viewer=True
#     )
#     _check_layer_initialization(viewer)
#     # ========================================================================
#     # # Automatic mask generation
#     # _autosegment_widget(v=viewer, min_object_size=30)
#     # # We expect four segmentation regions to be identified
#     # expected_segmentation_label_ids = np.array([0,1,2,3])
#     # np.testing.assert_equal(np.unique(viewer.layers["auto_segmentation"].data),
#     #                         expected_segmentation_label_ids)
#     viewer.close()  # must close the viewer at the end of tests


# def test_annotator_3d(make_napari_viewer_proxy, tmp_path):
#     """Integration test for annotator_2d widget with automatic mask generation.

#     * Creates 2D image embedding
#     * Opens annotator_2d widget in napari
#     * Test automatic mask generation
#     """
#     model_type = "vit_b"
#     embedding_path = tmp_path / "test-embedding.zarr"
#     # example data - a basic checkerboard pattern
#     image = np.zeros((16,16))
#     image[:8,:8] = 1
#     image[8:,8:] = 1

#     viewer = make_napari_viewer_proxy()
#     viewer = _initialize_viewer(image, None, None, None)  # TODO: fix hacky workaround
#     # test generating image embedding, then adding micro-sam dock widgets to the GUI
#     viewer = annotator_3d(
#         image,
#         embedding_path,
#         show_embeddings=False,
#         model_type=model_type,
#         v=viewer,
#         return_viewer=True
#     )
#     _check_layer_initialization(viewer)
#     # ========================================================================
#     # # Automatic mask generation
#     # _autosegment_widget(v=viewer, min_object_size=30)
#     # # We expect four segmentation regions to be identified
#     # expected_segmentation_label_ids = np.array([0,1,2,3])
#     # np.testing.assert_equal(np.unique(viewer.layers["auto_segmentation"].data),
#     #                         expected_segmentation_label_ids)
#     viewer.close()  # must close the viewer at the end of tests


# # def test_annotator_2d(make_napari_viewer_proxy, tmp_path):
# #     """Integration test for annotator_2d widget.

# #     * Creates 2D image embedding
# #     * Opens annotator_2d widget in napari
# #     * Test point prompts (add points, segment object, clear, and commit)
# #     * Test box prompt (add rectangle prompt, segment object, clear, and commit)
# #     ...
# #     """
# #     model_type = "vit_b"
# #     image = skimage.data.camera()
# #     embedding_path = tmp_path / "test-embedding.zarr"

# #     viewer = make_napari_viewer_proxy()
# #     viewer = _initialize_viewer(image, None, None, None)  # TODO: fix hacky workaround
# #     # test generating image embedding, then adding micro-sam dock widgets to the GUI
# #     viewer = annotator_2d(
# #         image,
# #         embedding_path,
# #         show_embeddings=False,
# #         model_type=model_type,
# #         v=viewer,
# #         return_viewer=True
# #     )
# #     _assert_initialization_is_correct(viewer)

# #     # ========================================================================
# #     # TEST POINT PROMPTS
# #     # Add three points in the sky region of the camera image
# #     sky_point_prompts = np.array([[70, 80],[50, 320],[80, 470 ]])
# #     viewer.layers["point_prompts"].data = sky_point_prompts

# #     # Segment sky region of image
# #     _segment_widget(v=viewer)  # segment slice
# #     # We expect all of the first 50 rows should be identified as sky,
# #     assert (viewer.layers["current_object"].data[0:50,:] == 1).all()
# #     # We also expect roughly 25% of the image to be sky
# #     sky_segmentation = np.copy(viewer.layers["current_object"].data)
# #     segmented_pixel_percentage = (np.sum(sky_segmentation == 1) / image.size) * 100
# #     assert segmented_pixel_percentage > 25
# #     assert segmented_pixel_percentage < 30

# #     # Clear segmentation current object and prompts
# #     _clear_widget(v=viewer)
# #     np.testing.assert_equal(viewer.layers["current_object"].data, 0)
# #     np.testing.assert_equal(viewer.layers["point_prompts"].data, 0)
# #     assert viewer.layers["prompts"].data == []  # shape data is list, not numpy array

# #     # Repeat segmentation and commit segmentation result
# #     viewer.layers["point_prompts"].data = sky_point_prompts
# #     _segment_widget(v=viewer)  # segment slice
# #     np.testing.assert_equal(sky_segmentation, viewer.layers["current_object"].data)
# #     # Commit segmentation
# #     _commit_segmentation_widget(v=viewer)
# #     np.testing.assert_equal(sky_segmentation, viewer.layers["committed_objects"].data)

# #     # ========================================================================
# #     # TEST BOX PROMPTS
# #     # Add rechangle bounding box prompt
# #     camera_bounding_box_prompt = np.array([[139, 254],[139, 324],[183, 324],[183, 254]])
# #     viewer.layers["prompts"].data = [camera_bounding_box_prompt]
# #     # Segment slice
# #     _segment_widget(v=viewer)  # segment slice
# #     # Check segmentation results
# #     camera_segmentation = np.copy(viewer.layers["current_object"].data)
# #     segmented_pixels = np.sum(camera_segmentation == 1)
# #     assert segmented_pixels > 2500  # we expect roughly 2770 pixels
# #     assert segmented_pixels < 3000  # we expect roughly 2770 pixels
# #     assert (camera_segmentation[150:175,275:310] == 1).all()  # small patch which should definitely be inside segmentation

# #     # Clear segmentation current object and prompts
# #     _clear_widget(v=viewer)
# #     np.testing.assert_equal(viewer.layers["current_object"].data, 0)
# #     np.testing.assert_equal(viewer.layers["point_prompts"].data, 0)
# #     assert viewer.layers["prompts"].data == []  # shape data is list, not numpy array

# #     # Repeat segmentation and commit segmentation result
# #     viewer.layers["prompts"].data = [camera_bounding_box_prompt]
# #     _segment_widget(v=viewer)  # segment slice
# #     np.testing.assert_equal(camera_segmentation, viewer.layers["current_object"].data)
# #     # Commit segmentation
# #     _commit_segmentation_widget(v=viewer)
# #     committed_objects = viewer.layers["committed_objects"].data
# #     # We expect two committed objects
# #     # label id 1: sky segmentation
# #     # label id 2: camera segmentation
# #     np.testing.assert_equal(np.unique(committed_objects), np.array([0, 1, 2]))
# #     np.testing.assert_equal(committed_objects == 2, camera_segmentation == 1)

# #     # ========================================================================
# #     viewer.close()  # must close the viewer at the end of tests





# # def test_something_else(make_napari_viewer_proxy):
# #     viewer = make_napari_viewer_proxy()
# #     # carry on with your test
# #     image = skimage.data.brick()
# #     viewer.add_image(image, name="raw")
# #     viewer.close()
