# ✈️ Flight Dynamic Pricing Model

A data-driven airline ticket pricing system that predicts ticket prices from flight-state variables and evaluates dynamic pricing strategies through scenario-based revenue simulation.

## 📌 Project Overview

Airline ticket prices change according to factors such as:

- Days remaining before departure
- Remaining seat availability
- Historical demand
- Competitor pricing
- Booking velocity
- Weekend/weekday status
- Flight capacity

This project develops a machine-learning-based pricing system that:

1. Validates and explores airline pricing data
2. Builds a baseline ticket-price prediction model
3. Compares linear and nonlinear models
4. Validates model stability using cross-validation
5. Tests feature dependency through ablation analysis
6. Generates dynamic ticket-price recommendations
7. Simulates revenue under different demand-response assumptions
8. Explains individual price recommendations through feature contributions

---

## 🧠 Problem Statement

Traditional static pricing assigns the same or predetermined price to a flight.

A dynamic pricing system instead adapts the recommended price according to the current state of the flight and market.

The objective of this project is to investigate whether available flight-state variables can be used to construct a predictive pricing model and a simulation framework for evaluating dynamic pricing decisions.

---

## 🏗️ System Architecture

```mermaid
flowchart TD

    A[Airline Dataset] --> B[Data Inspection]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Selection]

    D --> E[Linear Regression]
    D --> F[Random Forest]
    D --> G[Gradient Boosting]

    E --> H[Model Validation]
    F --> H
    G --> H

    H --> I[Dynamic Pricing Engine]

    I --> J[Recommended Ticket Price]

    J --> K[Demand Response Simulation]
    K --> L[Revenue Simulation]

    I --> M[Price Explainability]
    M --> N[Feature Contributions]