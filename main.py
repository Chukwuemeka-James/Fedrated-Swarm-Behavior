from src.train import train_federated

if __name__ == "__main__":
    # Path to your dataset
    data_path = 'Data/swarm_aligned'
    train_federated(data_path)
