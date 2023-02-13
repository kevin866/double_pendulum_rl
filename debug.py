import gym
import numpy as np
import keras 
import tensorflow as tf
env = gym.make('Acrobot-v1')

state_size = env.observation_space.shape[0]
print(env.reset()[0])
state = env.reset()[0]
print(np.reshape(state, [1, state_size]))
# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
# Create and train a new model instance.
model = create_model()

# Save the entire model as a SavedModel.
model.save('my_model.h5')