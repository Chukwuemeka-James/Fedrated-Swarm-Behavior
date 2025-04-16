import random
import tensorflow as tf
from src.utils import batch_data

def create_clients(data_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of datas and label lists.
        args: 
            data_list: a list of numpy arrays of training data
            label_list:a list of binarized labels for each data
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names: Example {}_{} = clients_1
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(data_list, label_list))
    random.shuffle(data)

    #shard the data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}

def process_and_batch_clients(clients, bs=32):
    #process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data, bs)
    return clients_batched
