import tensorflow as tf
from functions.utils import _obtain_input_shape

# https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

def resnet_identity_block(input_tensor, kernel_size, filters, stage, block, bias=False):
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
    conv3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"

    x = tf.keras.layers.Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "_bn")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, use_bias=bias, padding="same", name=conv3_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv3_name + "_bn")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_increase_name + "_bn")(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation("relu")(x)
    return x

def resnet_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
    conv1_proj_name = "conv" + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"

    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, use_bias=bias, name=conv1_reduce_name)(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "_bn")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same", use_bias=bias, name=conv3_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv3_name + "_bn")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_increase_name + "_bn")(x)

    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=bias, name=conv1_proj_name)(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_proj_name + "_bn")(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)
    return x

def RESNET50(include_top=True, weights="vggface", input_shape=None, pooling=None, classes=8631):
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    img_input = tf.keras.layers.Input(shape=input_shape)

    if tf.keras.backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    x = tf.keras.layers.Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding="same", name="conv1_7x7_s2")(img_input)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name="conv1_7x7_s2_bn")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = tf.keras.layers.AveragePooling2D((7, 7), name="avg_pool")(x)

    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(classes, activation="softmax", name="classifier")(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(img_input, x, name="vggface_resnet50")

    return model

def VGGFace(
    include_top=True, model="vgg16", weights="vggface", input_shape=None, pooling=None, classes=None
):
    if weights not in {"vggface", None}:
        raise ValueError
    
    if classes is None:
        classes = 8631

    if weights == "vggface" and include_top and classes != 8631:
        raise ValueError

    return RESNET50(
        include_top=include_top,
        input_shape=input_shape,
        pooling=pooling,
        weights=weights,
        classes=classes,
    )

def VGGFace_model(name_model = 'resnet50', shape = (224, 224, 3)):
    VGGFace_model = VGGFace(model=name_model, include_top=False, input_shape=shape, pooling='avg', weights=None)
    return VGGFace_model

def EE():
    basis_model = VGGFace_model()
    gaus = tf.keras.layers.GaussianNoise(0.1)(basis_model.output)
    x = tf.keras.layers.Dense(units=512, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation='relu', name='features')(gaus)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(7, activation='softmax')(x)
    bm = tf.keras.models.Model(basis_model.input, x)
    return bm
    
def load_weights_EE(path):
    EE_AN_model = EE()
    EE_AN_model.load_weights(path)
    EE_AN_model = tf.keras.models.Model(inputs=EE_AN_model.input, outputs=[EE_AN_model.get_layer('features').output])
    return EE_AN_model

def LSTM():
    input_lstm = tf.keras.Input(shape=(10, 512))
    X = tf.keras.layers.Masking(mask_value=0.)(input_lstm)
    X = tf.keras.layers.LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)
    X = tf.keras.layers.LSTM(256, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)
    X = tf.keras.layers.Dense(units=7, activation='softmax')(X)
    model = tf.keras.Model(inputs=input_lstm, outputs=X)
    return model

def load_weights_LSTM(path):
    LSTM_model = LSTM()
    LSTM_model.load_weights(path)
    return LSTM_model