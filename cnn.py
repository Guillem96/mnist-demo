import driftai as dai

import numpy as np
import tensorflow as tf

class CnnApproach(dai.RunnableApproach):

    @property
    def parameters(self):
        """
        Declare your parameters here
        """
        return [
            dai.parameters.IntParameter(initial=32, 
                                        limit=64,
                                        step=32,
                                        name='filters'),
            dai.parameters.FloatParameter(initial=1e-4,
                                          limit=1e-3,
                                          partitions=2,
                                          name='lr'),
            dai.parameters.IntParameter(initial=64,
                                        limit=128,
                                        step=64,
                                        name='clf_neurons')
        ]

    def learn(self, data, parameters):
        """
        Define, train and return your model here
        """
        X = np.array(data['X']).astype('float32') / 255.
        y = np.array(data['y']).astype(np.int)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(parameters['filters'], 5, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(parameters['clf_neurons'], activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(lr=parameters['lr']))
        model.fit(X, y, batch_size=32, epochs=1)
        return model # Return a trained model

    def inference(self, model, data):
        """
        Use the injected model to make predictions with the data
        """
        X = np.array(data['X']) / 255.
        preds = model.predict(X)
        return [ str(np.argmax(p)) for p in preds ]  # Return the prediction
