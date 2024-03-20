import tensorflow as tf
from tensorflow import keras

def wrap_tf_model(model):
    def model_jac(x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            pred = model(x_tensor)
            grads = t.gradient(pred, x_tensor)
        return grads.numpy()

    def model_forward(x):
        return model(x).numpy().squeeze()
    
    return model_forward, model_jac

def adult_NN_fit(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=512, epochs=50, verbose=0)

    return model

def compas_NN_fit(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=512, epochs=200, verbose=0)

    return model
