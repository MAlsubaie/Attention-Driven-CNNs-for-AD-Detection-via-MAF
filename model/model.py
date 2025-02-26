import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import GlobalAveragePooling3D, Dense, BatchNormalization, MaxPooling3D, Dropout, Concatenate
from tensorflow.keras.regularizers import l2

def channelwise_attention(inputs):
    avg_pool = GlobalAveragePooling3D()(inputs)
    attention = Dense(inputs.shape[-1], activation='sigmoid')(avg_pool)
    attention = layers.Reshape((1, 1, 1, inputs.shape[-1]))(attention)
    return layers.Multiply()([inputs, attention])

def build_3d_cnn(input_shape, l2_lambda=0.020):
    inputs_img = Input(shape=input_shape)

    # Block 1
    x = layers.Conv3D(32, (3, 3, 3), activation='gelu', kernel_regularizer=l2(l2_lambda))(inputs_img)
    x = channelwise_attention(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Block 2
    x = layers.Conv3D(64, (3, 3, 3), activation='gelu', kernel_regularizer=l2(l2_lambda))(x)
    x = channelwise_attention(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 3
    x = layers.Conv3D(128, (3, 3, 3), activation='gelu', kernel_regularizer=l2(l2_lambda))(x)
    x = channelwise_attention(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 4
    x = layers.Conv3D(256, (3, 3, 3), activation='gelu', kernel_regularizer=l2(l2_lambda))(x)
    x = channelwise_attention(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 5
    x = layers.Conv3D(512, (3, 3, 3), activation='gelu', kernel_regularizer=l2(l2_lambda))(x)
    x = channelwise_attention(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    gb1 = GlobalAveragePooling3D()(x)

    # Dense layers
    z1 = Dense(512, activation='gelu', kernel_regularizer=l2(l2_lambda))(gb1)
    z1 = Dropout(0.2)(z1)

    z2 = Dense(512, activation='silu', kernel_regularizer=l2(l2_lambda))(gb1)
    z2 = Dropout(0.2)(z2)

    z3 = Dense(512, activation='relu', kernel_regularizer=l2(l2_lambda))(gb1)
    z3 = Dropout(0.2)(z3)

    cat = Concatenate()([z1, z2, z3])
    z = Dropout(0.3)(cat)
    z = Dense(128, activation='gelu', kernel_regularizer=l2(l2_lambda))(z)

    outputs = Dense(1, activation='sigmoid')(z)

    model = models.Model(inputs=inputs_img, outputs=outputs)

    return model

def create_model(input_shape=None, load_pretrained=False, weights_path=None):
    if input_shape is None:
        input_shape = (128, 128, 128, 1)
    print("input_shape: ", input_shape)
    print("Building 3D CNN model...")
    model = build_3d_cnn(input_shape)
    print("3D CNN model built")

    print("*"*10,"Model Summary","*"*10)
    print(model.summary())
    print("*"*34)

    if load_pretrained:
        print("*"*34)
        print("Loading Pretrained Weights")
        if weights_path is None:
            print("Loading best weights")
            model.load_weights('./model/weights/best_weights.keras')
        else:
            print("Loading weights from: ", weights_path)
            model.load_weights(weights_path)

        print("*"*34)
    return model

