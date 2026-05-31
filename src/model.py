import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(inputs, num_filters):
    skip = conv_block(inputs, num_filters)
    pool = layers.MaxPooling2D((2, 2))(skip)
    return skip, pool


def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    bridge = conv_block(p4, 512)

    d1 = decoder_block(bridge, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = models.Model(inputs, outputs, name="U-Net")
    return model


model = build_unet()
model.summary()