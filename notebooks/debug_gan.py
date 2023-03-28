import tensorflow as tf

import pandas as pd
import numpy as np
from tensorflow import keras


df = pd.read_csv("data_clean/clean.csv")

column_names = df.columns
df = df.to_numpy()

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(df)
df = scaler.transform(df)

def build_generator(n_hidden=1, n_neurons=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu"))
        model.add(keras.layers.BatchNormalization())
    keras.layers.Dense(98, activation="sigmoid"),
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    return model
generator = build_generator()
# generator = keras.models.Sequential(
#     [
#         keras.layers.Flatten(),
#         keras.layers.Dense(300, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(2000, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(300, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(400, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(5000, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(500, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(400, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(300, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(200, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(100, activation="selu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dense(98, activation="sigmoid"),
#     ]
# )

discriminator = keras.models.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(3000, activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(2000, activation="selu"),
        keras.layers.BatchNormalization(),
        # keras.layers.Dense(300, activation="selu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(400, activation="selu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(5000, activation="selu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(500, activation="selu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(400, activation="selu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(300, activation="selu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(200, activation="selu"),
        # keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation="sigmoid"),  # not correct output dim
    ]
)

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
def train_gan(
    gan, dataset, n_epochs=100, n_noise=20000, accuracy=0.9
):

    """
    Inputs: 

    gan, this is a keras gan object made by combining two neural nets and
    restricting the trainability of one of them.

    dataset, this takes in regular tabular data. now this is training rowwise
    however i may change this to matrix wise like a picture.

    n_epochs, numper of times the gans go though training iterationations

    iterationations, number of times in gan iterationaton loop, 
    it would be a good idea to reduct this after the warmup period

    n_noise, this is the size of fake data generated

    
    Output:

    generators_saved, this is an iterationable list of keras objects that can be used
    
    discriminators_saved, same thing, these can be used to test

    for generator, discriminator in zip(gen, desc):
        noise = tf.random.normal(shape=dims)
        generated_data = generator(noise)
        judgement = discriminator(generated_data) # probs data is real
    """
    generator, discriminator = gan.layers
    generators_saved = []
    discriminators_saved = []
    log = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch} of {n_epochs}")

        mean_judgement = 0
        iteration = 0
        while mean_judgement < 0.9:
            iteration += 1
            random_index = np.random.permutation(len(dataset))
            X_batch = dataset[random_index, :]
            # phase 1 - training the discriminator

            mu = tf.random.uniform([1], minval=-0.1, maxval=0.1)
            sigma = tf.random.uniform([1], minval=0.9, maxval=1)
            noise = tf.random.normal(shape=(n_noise, 98),
                                     mean=mu,
                                     stddev=sigma) 

            generated_data = generator(noise)
            X_fake_and_real = tf.concat([generated_data, X_batch], axis=0)

            y1 = tf.constant([[0.0]] * noise.shape[0] + [[1.0]] * X_batch.shape[0])
            y1 = np.reshape(y1, (len(y1), 1))

            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            y2 = tf.constant([[1.0]] * n_noise)
            noise = tf.random.normal(shape=generated_data.shape)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

            generated_data = generator(noise)
            judgement = discriminator(generated_data) # probs data is real
            mean_judgement = np.mean(judgement)
            results = [iteration, np.mean(judgement), np.min(judgement), np.max(judgement)]
            print(
                "\n iteration", results[0],
                "\n mean", results[1],
                "\n max ", results[2],
                "\n min ", results[3]
                )
            log.append(results)
            
        generators_saved.append(generator)
        discriminators_saved.append(discriminator)
            
    return generators_saved, discriminators_saved


def generated_data_filter(gen, desc, threashold, dims=df.shape):
    """
    inputs
    gen, is the list of gans we wrote with the gan.ipynb

    desc, is the list of discriminators in the notebook gan.ipynb

    threashold, is what is the discriminator's predicted probability of the data being real
    we need to see to keep the data. 
    with a threashold = 0.99 we will drop every datapoint that the discriminator says has a 
    less than .99 change of being real. 
    we will need to play with this.

    """
    quality_data = np.empty((0, dims[1]), np.float32)
    for generator, discriminator in zip(gen, desc):
        noise = tf.random.normal(shape=dims)
        generated_data = generator(noise)
        judgement = discriminator(generated_data) # probs data is real
        data_fooling_discriminator = np.compress(np.ravel(judgement) > threashold, generated_data, axis=0)
        quality_data = np.append(quality_data, data_fooling_discriminator, axis=0)
    
    for discriminator in desc:
        judgement = discriminator(quality_data)
        print(np.mean(judgement))
        quality_data = np.compress(np.ravel(judgement) > threashold, quality_data, axis=0)
    return quality_data
 
generators_saved, discriminators_saved = train_gan(
    gan, df, n_epochs=12, n_noise=20000, accuracy=0.6
)
   
generated_dataset = generated_data_filter(generators_saved, discriminators_saved, threashold=0.03)

pd.DataFrame(generated_dataset, columns=column_names)