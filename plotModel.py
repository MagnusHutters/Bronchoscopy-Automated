import tensorflow as tf
import netron

# Load your model
model = tf.keras.models.load_model('pathModellabel.keras')
model = tf.keras.models.load_model('BronchoModel.keras')

# Save the model as a temporary file
model.save('BronchoModel.h5')

# Visualize the model with Netron
netron.start('BronchoModel.h5')