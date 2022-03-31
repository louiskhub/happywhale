def build_ds(self,imgage_paths,classes):
    global TARGET_SHAPE
    image_paths = TRAIN_DATA_PATH + "/" + imgage_paths

    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(classes, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(self.prepare_images_mapping, num_parallel_calls=8)
    return ds