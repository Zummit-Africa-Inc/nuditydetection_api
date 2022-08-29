# Import module
from nudenet import NudeClassifier

# initialize classifier (downloads the checkpoint file automatically the first time)
classifier = NudeClassifier()

# Classify single image
classifier.classify('image.jpg')
print("Downloaded Model!")
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}
# Classify multiple images (batch prediction)
# batch_size is optional; defaults to 4
#classifier.classify(['path_to_image_1', 'path_to_image_2'], batch_size=BATCH_SIZE)
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY},
#          'path_to_image_2': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}

# Classify video
# batch_size is optional; defaults to 4
#classifier.classify_video('path_to_video', batch_size=BATCH_SIZE)
# Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
#          "preds": {frame_i: {'safe': PROBABILITY, 'unsafe': PROBABILITY}, ....}}