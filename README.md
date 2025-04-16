# **Federated Swarm Behavior**

This project is dedicated to the **Colab ML Engineering Community** and learners around the world. It demonstrates a **Federated Learning (FL)** setup for training a machine learning model across multiple clients using a swarm-aligned dataset. The approach focuses on **decentralized training**, where each client trains the model with its own data and then contributes to the global model via aggregation.

---

## **Project Overview**

Federated Learning is an advanced machine learning technique that enables training across decentralized devices (clients) while keeping data local to each client. This project implements Federated Learning to train a simple **Multi-Layer Perceptron (MLP)** model across multiple clients, leveraging the `swarm_aligned.pkl` dataset. The experiment compares Federated Learning with centralized training as a baseline.

---

## **Dataset**

The **Swarm Behavior Dataset** is designed for analyzing and recognizing group activities, specifically the behaviors of **insects (flies)**. It is ideal for research in **collective behavior**, **group activity recognition**, and **video analysis**, particularly in biological and computational swarm intelligence contexts.

### **Description:**
This dataset contains videos of real flies exhibiting different types of swarm behaviors. The goal is to **classify the type of behavior** based on video features.

### **Data Details:**
- **Data Type**: Preprocessed numerical features (not raw video)  
- **Instances**: 240 samples  
- **Features**: 285 per instance (e.g., position, speed, interaction metrics, etc.)  
- **Classes**: 3 behavior types:
  - **Chasing**
  - **Following**
  - **Chain Formation**

Each instance represents a **10-second video clip** of multiple flies, encoded with spatial-temporal descriptors of their interactions.

### **Key Features:**
- **Position**: X and Y coordinates of each boid.
- **Velocity**: X and Y components of the velocity vector.
- **Alignment**: X and Y components of the alignment vector.
- **Separation**: X and Y components of the separation vector.
- **Cohesion**: X and Y components of the cohesion vector.
- **Neighborhood**: The number of boids within specific radii for alignment, separation, and cohesion.

Each instance consists of 2,400 features (12 attributes for each of 200 boids).

### **Class Labels:**
- **Flocking vs. Not Flocking**
- **Aligned vs. Not Aligned**
- **Grouped vs. Not Grouped**

### **Processed Dataset:**
Processed version: [`swarm_aligned.pkl`](https://drive.google.com/file/d/1WgNCTtNlVq7yjZdti0Fe85ELcBRclIGT/view?usp=sharing)

### **Source:**
Dataset hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour)

---

## **Directory Structure**

```
Federated Swarm Behavior/
│
├── data/                     
│   └── swarm_aligned.pkl     
│
├── src/                      
│   ├── model.py              
│   ├── utils.py              
│   ├── clients.py            
│   └── train.py              
│
├── docker/                   
│   └── Dockerfile            
│
├── main.py                   
├── Pipfile                   
├── Pipfile.lock              
└── README.md                 
```

---

## **Installation**

### **Install Dependencies Manually**

```bash
git clone https://github.com/yourusername/federated-mlp.git
cd federated-mlp
pip install pipenv
pipenv install
```

Place the `swarm_aligned.pkl` dataset inside the `data/` directory.

---

## **Running the Project**

```bash
python main.py
```

To override defaults:
```bash
python main.py --num_clients 5 --comms_round 5
```

---

## **Running with Docker**

**Build the image:**
```bash
docker build -t federated-swarm-mlp .
```

**Run the container (GPU enabled):**
```bash
docker run --gpus all -v $(pwd):/app federated-swarm-mlp
```

**Or without GPU:**
```bash
docker run -v $(pwd):/app federated-swarm-mlp
```

---

## **Acknowledgment**

This project draws inspiration from the Federated Learning walkthrough video by **Krish Naik**:
[Federated Learning Explained - YouTube](https://youtu.be/xZQL-i3SnFU?si=wl0Y_05MwNqlUgOt)