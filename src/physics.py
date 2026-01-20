import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. FÍSICA MATEMÁTICA ---

def get_watt_sample(a=0.988, b=2.249):
    """
    Sample neutron energies from the Watt Spectrum using rejection sampling 
    or inverse transform approximation.
    For this simulation, we use a simplified rejection method for accuracy.
    """
    # Rango típico de fisión [0, 15] MeV
    while True:
        val = np.random.uniform(0, 15)
        prob = np.exp(-a * val) * np.sinh(np.sqrt(b * val))
        
        # Max approx del Watt spectrum es alrededor de 0.4 para normalizar
        if np.random.random() < (prob / 0.4):
            return val # Retorna energía en MeV

# --- 2. VISUALIZACIÓN DE DATOS ---

def plot_results(history, save_path="results/simulation_result.png"):
    """
    Generate and save the neutron population graph.
    """
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Neutron Population', color='#1f77b4', linewidth=2)
    
    plt.axhline(y=history[0], color='r', linestyle='--', alpha=0.5, label='Initial Population')

    plt.title('Monte Carlo Criticality Simulation', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Number of Neutrons', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save
    plt.savefig(save_path, dpi=300)
    print(f" Graph successfully saved to: {save_path}")
