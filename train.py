import mlflow
import mlflow.tensorflow
import tensorflow as tf

mlflow.set_experiment("waste-classification")

with mlflow.start_run():

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224,224,3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # dummy example
    history = model.fit(X_train, y_train)

    mlflow.log_param("optimizer", "adam")
    mlflow.log_metric("accuracy", history.history["accuracy"][-1])

    mlflow.tensorflow.log_model(
        model,
        artifact_path="model"
    )