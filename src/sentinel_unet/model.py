"""Faithful reconstruction of the 2-level U-Net-like model in the lab notebook."""


def build_unet(input_shape=(256, 256, 12)):
    import tensorflow as tf

    def block(tensor, filters, stride=1):
        tensor = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        return tf.keras.layers.ReLU()(tensor)

    inputs = tf.keras.Input(shape=input_shape)
    skip1 = block(inputs, 32)
    x = block(skip1, 32, stride=2)
    skip2 = block(x, 64)
    x = block(skip2, 64, stride=2)
    x = block(x, 128)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = block(x, 64)
    x = tf.keras.layers.Concatenate()([x, skip2])
    x = block(x, 64)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = block(x, 32)
    x = tf.keras.layers.Concatenate()([x, skip1])
    x = block(x, 32)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="sentinel_unet")
