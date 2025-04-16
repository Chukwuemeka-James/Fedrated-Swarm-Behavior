# Federated Learning Project

This project is dedicated to the **Colab ML Engineering Community** and learners around the world. It demonstrates a Federated Learning (FL) setup for training a machine learning model across multiple clients using a swarm-aligned dataset. The approach focuses on decentralized training, where each client trains the model with its own data and then contributes to the global model via aggregation.

## Project Overview

Federated Learning is an advanced machine learning technique that enables training across decentralized devices (clients) while keeping data local to each client. This project implements Federated Learning to train a simple Multi-Layer Perceptron (MLP) model across multiple clients, leveraging the *swarm_aligned.pkl* dataset. The experiment compares Federated Learning with Centralized Training as a baseline.

## Dataset

The **Swarm Behavior Dataset** is designed for analyzing and recognizing group activities, specifically the behaviors of **insects (flies)**. It is ideal for research in **collective behavior**, **group activity recognition**, and **video analysis**, particularly in biological and computational swarm intelligence contexts. The dataset focuses on attributes like **Flocking vs. Not Flocking**, **Aligned vs. Not Aligned**, and **Grouped vs. Not Grouped**. The dataset captures the movement and interaction of 200 artificial boids (agents) in a 2D space.

### Description:
This dataset contains videos of real flies exhibiting different types of swarm behaviors. The goal is to **classify the type of behavior** based on video features.

### Data Details:

- **Data Type**: Preprocessed numerical features (not raw video)
- **Instances**: 240 samples
- **Features**: 285 per instance (e.g., position, speed, interaction metrics, etc.)
- **Classes**: 3 behavior types:
  - **Chasing**
  - **Following**
  - **Chain Formation**

Each instance represents a **10-second video clip** of multiple flies, encoded with spatial-temporal descriptors of their interactions.

### Key Features:
- **Position**: X and Y coordinates of each boid.
- **Velocity**: X and Y components of the velocity vector for each boid.
- **Alignment**: X and Y components of the alignment vector.
- **Separation**: X and Y components of the separation vector.
- **Cohesion**: X and Y components of the cohesion vector.
- **Neighborhood**: The number of boids within specific radii for alignment, separation, and cohesion.

Each instance consists of 2,400 features (12 attributes for each of 200 boids).

### Class Labels:
- **Flocking vs. Not Flocking**: Indicates whether the boids are flocking or not.
  - `1`: Flocking
  - `0`: Not Flocking
- **Aligned vs. Not Aligned**: Indicates whether the boids are aligned or not.
  - `1`: Aligned
  - `0`: Not Aligned
- **Grouped vs. Not Grouped**: Indicates whether the boids are grouped or not.
  - `1`: Grouped
  - `0`: Not Grouped

### Processed Dataset:
The processed version of the dataset, `swarm_aligned.pkl`, used for this project can be found [here](https://drive.google.com/file/d/1WgNCTtNlVq7yjZdti0Fe85ELcBRclIGT/view?usp=sharing).

### Source:
The dataset is hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour).

The dataset used for this project is the processed file containing aligned data shards, split among multiple clients, which are used in the federated setup.

## Directory Structure

The project directory is structured as follows:

```
Federated Swarm Behavior/
│
├── data/                     # Store your data file(s) here
│   └── swarm_aligned.pkl     # Dataset for model training
│
├── src/                      # All source code goes here
│   ├── model.py              # Contains the `MLP` class, which defines the architecture of the model (Multilayer Perceptron).
│   ├── utils.py              # Helper functions for loading data, batching, and scaling.
│   ├── clients.py            # Client creation and batching logic
│   └── train.py              # The main training logic that handles Federated and SGD training loops.
│
├── docker/                   # Docker-related files
│   └── Dockerfile            # Dockerfile for containerized environment (uses pipenv)
│
├── main.py                   # The entry point for running the full training pipeline.
├── Pipfile                   # Pipenv file listing project dependencies
├── Pipfile.lock              # Lockfile to ensure reproducible builds
└── README.md                 # Project overview
```

## Installation

### Install Dependencies Manually

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/federated-mlp.git
   cd federated-mlp
   ```

2. **Install Pipenv**:
     ```bash
      pip install pipenv
     ```

3. Install the required dependencies:
     ```bash
     pipenv install
     ```

4. Download or prepare your dataset and place it in the `data/` directory. Ensure the dataset is in the format that the code can process (e.g., a pickle file, like `swarm_aligned.pkl`).

### Running the Project

By default, the training will be run with **10 clients** and **10 communication rounds**. To run the training with these default values, use the following command:

```bash
python main.py
```

If you'd like to specify the number of clients and communication rounds, you can override the defaults by running:

```bash
python main.py --num_clients 5 --comms_round 5
```

In this example, the model will be trained with **5 clients** and **5 communication rounds**.

### Running with Docker

To run this project in a Docker container, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t Federated-Swam-MLP
   ```

2. **Run the Docker container**:

   - **With GPU support** (if you have a GPU):
     ```bash
     docker run --gpus all -v $(pwd):/app Federated-Swam-MLP
     ```
     - The `--gpus all` flag enables GPU support (make sure you have Docker configured to use GPUs, such as having NVIDIA Docker installed).

   - **Without GPU support** (if you don't have a GPU):
     ```bash
     docker run -v $(pwd):/app Federated-Swam-MLP
     ```
     - The `-v $(pwd):/app` command mounts your current directory to the container, ensuring that all files are accessible inside the container.
     - This will run the project using your CPU instead of the GPU.

3. **Monitor the training process**: Once the container is running, the training will begin. You can monitor the logs directly in your terminal window.