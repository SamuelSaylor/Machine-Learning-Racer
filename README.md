# 🏁 Machine Learning Racer

## Meet the Team
Matthew Beck, Samuel Saylor, Samantha Machado, Prince Patel  
Built for the UTSA Code Quantum Hackathon 🚀  
*Created in full after eight hours of labor.*

---

> **"Race, Learn, Repeat: Watch AI master the track while you challenge it yourself!"**

---

## 🧠 Project Overview

This project is a **reinforcement learning-powered racing simulator** where an AI agent learns how to navigate a track as efficiently as possible.

Using trial-and-error learning, the agent improves over time by:
- Maximizing speed and forward progress  
- Learning optimal racing lines  
- Avoiding penalties like collisions or leaving the track  

An **optional single-player mode** allows users to race against the trained AI in real time.

This project demonstrates how machine learning can be applied to dynamic, physics-based environments by combining game development with AI training.

---

## ⚙️ Features

### 🏎️ AI-Controlled Car
- Trained using Proximal Policy Optimization (PPO)
- Learns driving behavior from scratch (no hardcoded pathing)

### 🎮 Single-Player Mode
- Human vs AI racing  
- Real-time keyboard controls  
- Same physics system for both player and AI  

### 🧩 Reinforcement Learning System

**Positive Reinforcement:**
- Moving forward efficiently  
- Staying on track  
- Passing checkpoints  
- Completing laps quickly  

**Negative Reinforcement:**
- Driving off track  
- Entering deadzones (forced respawn)  
- Reversing unnecessarily  
- Excessive steering or spinning  

---

## 🗺️ Track & Physics System

- Pixel-based collision using masks  
- Separate layers:
  - Track boundary  
  - Deadzone (crash areas)  
  - Cosmetic background  
- Realistic car physics:
  - Acceleration  
  - Friction  
  - Turning dynamics  

---

## 🧪 Technical Breakdown

### 📁 `racing_env.py` (Core RL Environment)

Defines the Gymnasium-compatible environment used for training.

**Responsibilities:**
- Observation space (16 features):
  - Speed  
  - Steering input  
  - Raycast distances (track awareness)  
  - Track state (on/off/deadzone)  
  - Checkpoint progress  

- Raycasting system:
  - 9 directional sensors simulate vision  
  - Helps the AI understand track geometry  

- Handles:
  - Reward shaping  
  - Collision detection  
  - Checkpoint progression  
  - Respawning logic  

👉 This is where the AI learns how to drive.

---

### 🤖 `machinelearning.py` (Training Pipeline)

Handles training using PPO.

**Key Features:**
- Parallel environments for faster learning  
- Domain randomization:
  - Slight variation in car stats each run  
  - Prevents overfitting  

- Configurable parameters:
  - `--n-envs` (parallel environments)  
  - `--timesteps` (training duration)  
  - `--physics-substeps` (simulation accuracy)  

---

## ⚡ Training Settings

- Learning Rate: `0.0003`  
- Gamma: `0.99`  
- PPO-based training with Stable-Baselines3  

---

## ⚠️ Training Challenges

### Problem: Car not staying on track

Early in training, the agent often:
- Spins in circles  
- Moves backward  
- Gets stuck exploiting small reward signals  

### Solutions:
- Added forward velocity reward  
- Penalized:
  - Excessive steering  
  - Spinning (yaw penalty)  
  - Reverse movement  
- Introduced checkpoint-based rewards  
- Used domain randomization for robustness  

---

## 📊 Tech Stack

- Python 3.12+  
- Pygame (graphics & input)  
- Stable-Baselines3 (reinforcement learning)  
- Gymnasium (environment interface)  
- NumPy (numerical operations)  
- Matplotlib (training visualization)  

---
