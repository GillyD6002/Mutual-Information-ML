import configparser
import ast
import math

import tensorflow as tf
from tensorflow import keras as ks
import numpy as np

from src import image as img

# This module runs the MI estimation experiments.

class Model():

    # This class holds the fully-connected neural network
    # that is used for MI estimation.

    def __init__(self, image_shape, settings):

        # The model can be configured based on the numuber of 
        # layers, learning rate, optimizers, and dropout.

        self.drop = float(settings['drop'])
        self.learn_rate = float(settings['learn'])
        self.layers = ast.literal_eval(settings['layers'])
        self.patience = int(settings['patience'])
        self.optimizer = settings['optm']
        self.build_model(image_shape)

    def build_model(self, image_shape):

        # The model is constructed such that it contains two inputs
        # one corresponding to the normal images and another to
        # the mixed images. These inputs generate distinct outputs,
        # but are processed by the same model layers.

        joint_input = ks.Input(shape = image_shape)
        marginal_input = ks.Input(shape = image_shape)
        model_core = ks.models.Sequential()
        model_core.add(ks.Input(shape = image_shape))
        model_core.add(ks.layers.Flatten())
        for layer_size in self.layers:
            model_core.add(ks.layers.Dense(layer_size, activation = 'relu'))
            model_core.add(ks.layers.Dropout(self.drop))
        model_core.add(ks.layers.Dense(1, activation = None))
        joint_output = model_core(joint_input)
        marginal_output = model_core(marginal_input)
        self.model = ks.Model(inputs = [joint_input, marginal_input], outputs = [joint_output, marginal_output])
        
    def compile_model(self, loss_functions):

        # The model is compiled based on the specified opimizer
        # and learning rate.

        if self.optimizer == 'adam':
            optimizer = ks.optimizers.Adam(learning_rate = self.learn_rate)
        elif self.optimizer == 'rms':
            optimizer = ks.optimizers.RMSprop(learning_rate = self.learn_rate)
        elif self.optimizer == 'sgd':
            optimizer = ks.optimizers.SGD(learning_rate = self.learn_rate)
        else:
            raise ValueError("Optimizer not recognized.")
        self.model.compile(
            optimizer = optimizer,
            loss = loss_functions)

    def train(self, train_itr, val_itr, train_steps, val_steps, epochs):

        # The model is trained using early stoppage according to the
        # patience setting.

        self.model.fit(
            train_itr,
            steps_per_epoch = train_steps,
            epochs = epochs,
            validation_data = val_itr,
            validation_steps = val_steps,
            callbacks =  [tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss', min_delta = 0, patience = self.patience, restore_best_weights = True)],
            verbose = 1
        )
    
    def evaluate_MI(self, image_iterator, num_steps):

        # This function uses the trained model to estimate the
        # MI of a given image set.

        cum_joint = 0
        cum_marginal = 0
        for (count, (image_batch, _)) in enumerate(image_iterator):
            print('\rCount: {}'.format(count), end = '')
            [joint_outputs, marginal_outputs] = self.model.predict_on_batch(image_batch)
            cum_joint += np.mean(joint_outputs)
            cum_marginal += np.mean(np.exp(marginal_outputs))
            if count >= num_steps:
                break
        print('')
        est_mi = cum_joint / num_steps - np.log(cum_marginal / num_steps)
        direct_mi = cum_joint / num_steps
        return (est_mi, direct_mi)

class LogsiticRegression(Model):

    # This model uses the cross-entropy as its loss function.

    def __init__(self, image_shape, settings):
        Model.__init__(self, image_shape, settings)
        loss_functions = [logistic_loss_joint, logistic_loss_marginal]
        self.compile_model(loss_functions)

# Backwards-compatible alias for the (historically misspelled) class name.
# Existing code and notebooks reference `LogsiticRegression`; new code can use
# the correctly spelled `LogisticRegression`.
LogisticRegression = LogsiticRegression

class MINE(Model):

    # This model uses the MINE loss function.

    def __init__(self, image_shape, settings):
        Model.__init__(self, image_shape, settings)
        loss_functions = [biased_MINE_loss_joint, biased_MINE_loss_marginal]
        self.compile_model(loss_functions)

class Index():

    # This class is used to conveniently construct randomized indices.

    def __init__(self, num_indicies):
        self.indices = np.random.permutation(np.arange(num_indicies))

    def draw(self, size):
        choice = self.indices[:size]
        self.indices = self.indices[size:]
        return choice

def get_mixed_indices(num_indices):

    # This function generates randomized pairings drawn with
    # replacement from a set of indices, and avoids duplicate
    # pairings.

    index_1 = Index(num_indices)
    index_2 = Index(num_indices)
    dupl_positions = np.nonzero(np.equal(index_1.indices, index_2.indices))[0]
    while dupl_positions.size != 0:
        dupl_indices = index_1.indices[dupl_positions]
        random_positions_1 = np.random.choice(num_indices, dupl_positions.size, replace = False)
        random_positions_2 = np.random.choice(num_indices, dupl_positions.size, replace = False)
        random_indices_1 = index_1.indices[random_positions_1]
        random_indices_2 = index_2.indices[random_positions_2]
        index_1.indices[dupl_positions] = random_indices_1
        index_1.indices[random_positions_1] = dupl_indices
        index_2.indices[dupl_positions] = random_indices_2
        index_2.indices[random_positions_2] = dupl_indices
        dupl_positions = np.nonzero(np.equal(index_1.indices, index_2.indices))[0]
    return (index_1, index_2)

def logistic_loss_joint(unused_y_true, joint_output):

    # This function is the cross-entropy loss function with
    # labels of value 1.

    labels = tf.ones_like(joint_output)
    logits = joint_output
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(losses)
    return loss

def logistic_loss_marginal(unused_y_true, marginal_output):

    # This function is the cross-entropy loss function with
    # labels of value 0.

    labels = tf.zeros_like(marginal_output)
    logits = marginal_output
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(losses)
    return loss

def biased_MINE_loss_joint(unused_y_true, joint_output):

    # This function computes the joint portion of the MINE loss.

    loss = -tf.reduce_mean(joint_output)
    return loss

def biased_MINE_loss_marginal(unused_y_true, marginal_output):

     # This function computes the marginal portion of the MINE loss.

    avg_marginal_exp = tf.reduce_mean(tf.exp(marginal_output))
    loss = tf.math.log(avg_marginal_exp)
    return loss

def get_finite_dataset(images, inner_region, batch_size, loop = True):

    # This function returns a generator that takes a set of images and
    # generates joint sample (image left alone) and marginal samples
    # (center part of image swapped with another image) to use for
    # MI estimation. 

    num_batches = math.ceil(images.shape[0] / batch_size)
    (top, bottom, left, right) = inner_region
    rand = np.random.RandomState()
    if loop:
        itr = iter(int, 1) # Infinite iterator
    else:
        itr = range(1)
    for _ in itr:
        image_indices = Index(images.shape[0])
        (mixed_inner_indices, mixed_outer_indices) = get_mixed_indices(images.shape[0])
        for _ in range(num_batches):
            image_choice = image_indices.draw(batch_size)
            mixed_inner_choice = mixed_inner_indices.draw(batch_size)
            mixed_outer_choice = mixed_outer_indices.draw(batch_size)
            mixed_images = images[mixed_outer_choice]
            mixed_images[:, top:bottom, left:right] = images[mixed_inner_choice][:, top:bottom, left:right]
            joint_images = images[image_choice]
            yield ((joint_images, mixed_images), (np.zeros(joint_images.shape[0]), np.zeros(mixed_images.shape[0])))

def cycle_generator(generator_fn, *args, **kwargs):

    # Repeatedly re-invokes a finite generator factory, yielding a true
    # generator object (Keras's data adapters reject itertools.cycle,
    # which is not a generator instance).

    while True:
        yield from generator_fn(*args, **kwargs, loop = False)

def run_bipartition(inner_length, alg_settings, param_settings, eval_steps = 5000):

    # This function runs the MI estimation experiment using the
    # provided settings and partition size. eval_steps controls how many
    # validation batches evaluate_MI averages over; the default of 5000
    # matches the paper's full-scale runs, but a smaller value is more
    # appropriate when num_images has been reduced for a quicker demo.

    num_images = max(1, int(alg_settings["num_images"]))
    (images, _, _) = img.get_images(
        alg_settings["image_type"], 
        num_images, 
        strength = alg_settings["strength"])
    (_, height, width) = images.shape
    images = np.expand_dims(images, axis = 3)
    inner_region = img.get_center_region(inner_length, height, width)

    algorithm = alg_settings["algorithm"]
    if algorithm == 'mine':
        net = MINE(images.shape[1:], param_settings)
    elif algorithm == 'logistic':
        net = LogisticRegression(images.shape[1:], param_settings)
    else:
        raise ValueError('Algorithm {} not recognized.'.format(algorithm))

    val_start = int(images.shape[0] * float(param_settings['val']))
    train_images = images[val_start:]
    val_images = images[:val_start]
    batch_size = int(param_settings['batch'])
    
    train_steps = int(np.ceil(train_images.shape[0] / batch_size))
    val_steps = int(np.ceil(val_images.shape[0] / batch_size))

    train_itr = get_finite_dataset(train_images, inner_region, batch_size, loop = True)
    val_itr = cycle_generator(get_finite_dataset, val_images, inner_region, batch_size)

    net.train(train_itr, val_itr, train_steps, val_steps, int(param_settings['epoch']))
    (est_mi, direct_mi) = net.evaluate_MI(val_itr, eval_steps)
    return [est_mi, direct_mi]

if __name__ == "__main__":

    # The following code loads settings from the alg.ini and
    # mine.ini configuration files, and then runs the specified
    # experiments.

    alg_parser = configparser.ConfigParser()
    alg_parser.read("alg.ini")
    alg_settings = alg_parser["alg"]

    param_parser = configparser.ConfigParser()
    param_parser.read("mine.ini")
    param_settings = param_parser["dense"]

    algorithm = alg_settings["algorithm"]
    image_type = alg_settings["image_type"]
    strength = alg_settings["strength"]
    start_length = int(alg_settings["start_length"])
    num_images = int(alg_settings["num_images"])

    if image_type in img.rho_values.keys():
        rho = img.rho_values[image_type][strength]
    else:
        rho = 0

    max_length = 28
    if start_length < 0:
        MIs = []
        for length in range(abs(start_length), max_length):
            print("Length: {}".format(length))
            mi_pair = run_bipartition(length, alg_settings, param_settings)
            MIs.append(mi_pair)
            print("\nMI Lower: {} | MI Direct: {}".format(*mi_pair))
            print("{} - {} - {} - {}".format(algorithm, image_type, rho, num_images))
    else:
        mi = run_bipartition(start_length, alg_settings, param_settings)
        print(mi)
