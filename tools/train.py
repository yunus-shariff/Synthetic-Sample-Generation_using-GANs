import tensorflow as tf

import numpy as np
from tensorflow import keras

def build_network(output_dim, n_hidden, n_neurons, learning_rate):

    """Assembles the neural network architecture.
    Parameters
    -------
    output_dim : array
        this is the dimension we wish the output to be
        Case 1 Generator output (n, n_columns) of data we wish to generate
        Case 2 Discriminator output (n, 1) probability vector p(data_real|data)

    n_hiden : int 
        number of layers of the neural net

    n_neurons : int 
        number of neuros in the network

    learning_rate : float
        learning rate through the network 

    Returns
    -------
    This outputs a keras neural net ready to be trained.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu"))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(output_dim + 3, activation="selu"))  
    model.add(keras.layers.Dense(output_dim, activation="sigmoid"))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    return model

def train_gan(
    generator, discriminator, dataset, n_epochs=100, n_noise=20000
):
    """This function trains the GAN
    Parameters
    -------- 
    generator : keras object
        input an assembled keras neural network. It is critical that
        its output is the same width but can be any length (n, n_cols) 
        format as the data going in. 
    
    discriminator : keras object
        input an assembeld keras neural network. This must have a single
        value of an output of the same length as the generator. (n, 1)

    dataset : numpy array
        This takes an (n_row, n_col) numpy array. This is the data that
        you wish to generate more samples of

    n_epochs : int
        this is the number of times the network trains before outputting a
        result

    n_noise : int
        This is the number of generated samples to make, as it will be just
        transformed noise.
    Returns
    -------
    numpy array
        this is the generated datasamples
    """

    gan = keras.models.Sequential([generator, discriminator])
  
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    generator, discriminator = gan.layers
    data_out = np.empty((0, dataset.shape[1]))


    for epoch in range(250):
        np.random.seed(epoch)
        random_index = tf.random.uniform(shape=(n_noise,), minval=0, maxval=len(dataset), dtype=tf.int32)
        X_batch = dataset[random_index, :]
        for iteration in range(5):

            noise = tf.random.normal(shape=X_batch.shape,
                                     mean=0,
                                     stddev=1) 

            generated_data = generator(noise)
            X_fake_and_real = tf.concat([generated_data, X_batch], axis=0)
            y1 = tf.concat([tf.zeros(n_noise), tf.ones(n_noise)], axis=0)
            
            # training discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # training the generator

            noise = tf.random.normal(shape=X_batch.shape,
                                     mean=0,
                                     stddev=1) 
            
            discriminator.trainable = False
            gan.train_on_batch(noise, tf.ones(n_noise))
         
        generated_data = generator(noise)
        rand = tf.random.uniform(shape=(1,), minval=0, maxval=X_batch.shape[0], dtype=tf.int32)

        data_out = np.concatenate([data_out, generated_data[ :1 , :]])
    
    return data_out
    
