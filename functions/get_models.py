import tensorflow as tf
from keras_vggface.vggface import VGGFace

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