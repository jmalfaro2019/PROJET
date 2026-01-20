# main.py
import matplotlib.pyplot as plt
from src.simulation import run_monte_carlo
from src.physics import plot_results

def main():
    print("Starting Monte Carlo Simulation...")
    
    # Configuración
    n_neutrons = 1000
    n_generations = 500
    
    # Ejecutar lógica (importada de src/)
    history = run_monte_carlo(n_neutrons, n_generations)
    
    # Graficar y guardar
    plot_results(history, save_path="results/simulation_result.png")
    print("Simulation complete. Results saved in 'results/'")

if __name__ == "__main__":
    main()