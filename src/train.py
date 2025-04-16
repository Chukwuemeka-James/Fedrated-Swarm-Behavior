import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # added for learning rate scheduling
from src.model import SimpleMLP
from src.utils import load, weight_scalling_factor, scale_model_weights, sum_scaled_weights, test_model
from src.clients import create_clients, process_and_batch_clients
import numpy as np
import random

def train_federated(data_path, comms_round=10, num_clients=10):
    data_list, label_list = load(data_path)
    labels = list(set(label_list.tolist()))  # Unique labels

    # It calculates the number of unique classes (labels) in your dataset
    n_values = np.max(label_list) + 1

    # Convert labels to one-hot encoded format
    label_list = np.eye(n_values)[label_list]

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.1, random_state=42)

    # Create clients
    clients = create_clients(X_train, y_train, num_clients=num_clients, initial='client')
    clients_batched = process_and_batch_clients(clients)

    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

    # Learning rate schedule
    initial_lr = 0.01
    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True
    )

    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # Initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(data_list.shape[1], len(labels))
    optimizer_global = SGD(learning_rate=lr_schedule, momentum=0.9)
    global_model.compile(loss=loss, optimizer=optimizer_global, metrics=metrics)

    # Begin federated learning loop
    for comm_round in range(comms_round):
        global_weights = global_model.get_weights()
        scaled_local_weight_list = []

        client_names = list(clients_batched.keys())
        random.shuffle(client_names)

        for client in tqdm(client_names, desc='Progress Bar'):
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(data_list.shape[1], len(labels))

            # Create a fresh optimizer for each new local model
            local_optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)

            local_model.compile(loss=loss, optimizer=local_optimizer, metrics=metrics)
            local_model.set_weights(global_weights)
            local_model.fit(clients_batched[client], epochs=1, verbose=0)

            scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            tf.keras.backend.clear_session()

        average_weights = sum_scaled_weights(scaled_local_weight_list)
        global_model.set_weights(average_weights)

        for (X_test, Y_test) in test_batched:
            global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

    # Centralized training baseline
    SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
    smlp_SGD = SimpleMLP()
    SGD_model = smlp_SGD.build(data_list.shape[1], len(labels))
    SGD_model.compile(loss=loss, optimizer=SGD(learning_rate=lr_schedule, momentum=0.9), metrics=metrics)
    SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

    for (X_test, Y_test) in test_batched:
        SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)
