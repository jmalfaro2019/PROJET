import numpy as np
from src.material import MaterialU235, MaterialU238, Neutron 
from src.physics import get_watt_sample

def run_monte_carlo(n_initial_neutrons=1000, 
                n_generations_max=1000, 
                n_max_neutrons=10000,
                N_U235=0.007, 
                N_U238=0.993):
    
    u235 = MaterialU235()
    u238 = MaterialU238()
    
    neutrons = []
    # Initialize thermal neutrons
    for i in range(n_initial_neutrons):
        neutrons.append(Neutron(energy=0.025)) 
        
    num_neutrons_history = []
    generation = 0
    
    while generation <= n_generations_max:
        current_count = len(neutrons)
        print(f"Gen {generation}: Neutrons = {current_count}")
        num_neutrons_history.append(current_count)
        
        if current_count == 0:
            print("Reaction stopped.")
            break
        if current_count > n_max_neutrons:
            print("Reaction exploded.")
            break
            
        new_neutrons = []
        
        for neutron in neutrons:
            E = neutron.get_energy()
            
            # 1. Macroscopic probability logic
            s_tot_235 = u235.get_total_sigma(E)
            s_tot_238 = u238.get_total_sigma(E)
            
            macro_235 = s_tot_235 * N_U235
            macro_238 = s_tot_238 * N_U238
            total_macro = macro_235 + macro_238
            
            # Hit U-235 or U-238?
            if np.random.random() <= (macro_235 / total_macro):
                mat = u235
            else:
                mat = u238

            # 2. Microscopic Physics (Probabilities from Cross Sections)
            s_fis = mat.get_sigma_fission(E)
            s_cap = mat.get_sigma_capture(E)
            s_ine = mat.get_sigma_inelastic(E) # New method
            s_tot = mat.get_total_sigma(E)
            
            # Normalize probabilities
            p_fis = s_fis / s_tot
            p_cap = s_cap / s_tot
            p_ine = s_ine / s_tot
            # p_ela is the remainder
            
            roll = np.random.random()
            
            # === CASE A: FISSION ===
            if roll <= p_fis:
                # Determine Nu
                if E <= 1.0: n_mean = 2.43
                else:        n_mean = 2.50
                
                n_produced = int(n_mean) + 1 if np.random.random() < (n_mean - int(n_mean)) else int(n_mean)
                
                for _ in range(n_produced):
                    # IMPORTANTE: Asegúrate de que get_watt_sample esté importado
                    new_neutrons.append(Neutron(get_watt_sample() * 1e6))
            
            # === CASE B: CAPTURE ===
            elif roll <= (p_fis + p_cap):
                pass # Neutron dies
            
            # === CASE C: INELASTIC SCATTERING (New Logic) ===
            elif roll <= (p_fis + p_cap + p_ine):
                new_E = np.random.uniform(50000, 400000)
                if new_E >= E:
                    new_E = E * 0.9 
                
                neutron.set_energy(new_E)
                new_neutrons.append(neutron)
                
            # === CASE D: ELASTIC SCATTERING ===
            else:
                factor = np.random.uniform(0.98, 1.0)
                neutron.set_energy(E * factor)
                new_neutrons.append(neutron)

        neutrons = new_neutrons
        generation += 1

    # Devuelve la historia para poder graficarla
    return num_neutrons_history