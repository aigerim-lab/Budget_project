from tensorflow.keras.utils import plot_model
import tensorflow as tf

model = tf.keras.models.load_model("models/risk_model.keras", compile=False)

plot_model(
    model,
    to_file="models/model_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

print("Model diagram saved to models/model_architecture.png")