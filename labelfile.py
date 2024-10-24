import tensorflow as tf
from tensorflow import keras

from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image_dataset_from_directory as im_ds_from_dir

def ret_model(path='enc10_3.keras'):
    model = keras.models.load_model('enc10_3.keras')

    model.summary()
    cluster = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name='conv2d_7').output
    )

    return cluster

def load_dir(dir):
    image_dir=dir
    target_size=(32,32) # set this to the target size you want
    image_gen=im_ds_from_dir(image_dir, color_mode='grayscale',image_size=target_size,
                            labels=None, shuffle=False)
    
    norm_layer = tf.keras.layers.Rescaling(1./255)
    image_gen_map = image_gen.map(lambda x: (norm_layer(x) - 1e-6))
    return image_gen.file_paths ,image_gen_map


def ret_predict(cluster,image_gen,db_len, n_cluster=3):
    bottleneck_features = cluster.predict(image_gen)

    estimator = KMeans(n_clusters=n_cluster)

    estimator.fit(bottleneck_features.reshape(db_len,-1))

    return estimator.labels_

def ret_labels(dir, model_path='enc10_3.keras', ncluster=3):
    model = ret_model(model_path)
    paths, img_gen = load_dir(dir)
    labels = ret_predict(model, img_gen, ncluster)
    return paths, labels