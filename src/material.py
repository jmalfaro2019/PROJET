import numpy as np
from scipy.interpolate import interp1d

class Neutron:
    def __init__(self, energy=0.025):
        """
        Initialize a neutron.
        energy: Energy in eV (default thermal 0.025 eV)
        """
        self.energy = energy

    def get_energy(self):
        return self.energy

    def set_energy(self, E):
        self.energy = E

class MaterialU235:
    def __init__(self):
        # --- Data Based on ENDF/B-VIII ---
        self.energy_grid = np.array([
            1e-5,     0.0253,   0.1,      1.0,      10.0,
            100.0,    1000.0,   1e4,      1e5,      1e6,
            2e6,      10e6,     20e6
        ])

        # Total Cross-section [Barns]
        self.sigma_total_grid = np.array([
            2500.0,   698.0,    280.0,    100.0,    60.0,
            35.0,     20.0,     15.0,     10.0,     7.5,
            7.2,      7.5,      8.0
        ])
        
        # Fission [Barns]
        self.sigma_fission_grid = np.array([
            2100.0,   584.0,    200.0,    40.0,     20.0,
            15.0,     7.0,      2.5,      1.5,      1.2,
            1.3,      2.2,      2.0
        ])

        # Capture (n, gamma) [Barns]
        self.sigma_capture_grid = np.array([
            350.0,    99.0,     40.0,     10.0,     5.0,
            4.0,      3.0,      1.5,      0.6,      0.1,
            0.05,     0.01,     0.01
        ])

        # --- NEW: Inelastic Scattering [Barns] ---
        # Threshold around ~10-40 keV. Below that is 0.
        self.sigma_inelastic_grid = np.array([
            0.0,      0.0,      0.0,      0.0,      0.0,
            0.0,      0.0,      0.0,      0.5,      1.5,  # Significant at high energy
            1.8,      2.0,      2.0
        ])

        # Interpolators
        self.interpolator_total = interp1d(np.log(self.energy_grid), np.log(self.sigma_total_grid), kind='linear', fill_value="extrapolate")
        self.interpolator_fission = interp1d(np.log(self.energy_grid), np.log(self.sigma_fission_grid), kind='linear', fill_value="extrapolate")
        self.interpolator_capture = interp1d(np.log(self.energy_grid), np.log(self.sigma_capture_grid), kind='linear', fill_value="extrapolate")
        
        # FIX: To avoid log(0), we replace 0.0 with a tiny number 1e-20 for interpolation setup
        safe_inelastic = np.where(self.sigma_inelastic_grid == 0, 1e-20, self.sigma_inelastic_grid)
        self.interpolator_inelastic = interp1d(np.log(self.energy_grid), np.log(safe_inelastic), kind='linear', fill_value="extrapolate")

    def get_total_sigma(self, energy_ev):
        return np.exp(self.interpolator_total(np.log(energy_ev)))
    
    def get_sigma_fission(self, energy_ev):
        return np.exp(self.interpolator_fission(np.log(energy_ev)))

    def get_sigma_capture(self, energy_ev):
        return np.exp(self.interpolator_capture(np.log(energy_ev)))

    def get_sigma_inelastic(self, energy_ev):
        # If energy is below threshold (~40 keV), return 0 directly
        if energy_ev < 40000:
            return 0.0
        return np.exp(self.interpolator_inelastic(np.log(energy_ev)))

    def get_sigma_elastic(self, energy_ev):
        """
        Sigma_Elastic = Total - (Fission + Capture + Inelastic)
        """
        st = self.get_total_sigma(energy_ev)
        sf = self.get_sigma_fission(energy_ev)
        sc = self.get_sigma_capture(energy_ev)
        si = self.get_sigma_inelastic(energy_ev)
        return max(0.0, st - (sf + sc + si))


class MaterialU238:
    def __init__(self):
        # --- Data Based on ENDF/B-VIII ---
        self.energy_grid = np.array([
            1e-5,     0.0253,   0.1,      1.0,      10.0,
            100.0,    1000.0,   1e4,      1e5,      1e6,
            2e6,      10e6,     20e6
        ])

        # Total [Barns]
        self.sigma_total_grid = np.array([
            400.0,    12.0,     10.5,     15.0,     25.0,
            20.0,     15.0,     13.0,     10.0,     7.5,
            7.2,      7.5,      7.8
        ])
        
        # Fission [Barns]
        self.sigma_fission_grid = np.array([
            1e-9,     1e-9,     1e-9,     1e-9,     1e-9,
            1e-9,     1e-9,     1e-5,     1e-3,     0.05,
            0.55,     1.0,      1.2
        ])

        # Capture (n, gamma) [Barns]
        self.sigma_capture_grid = np.array([
            6.0,      2.68,     1.5,      0.5,      20.0, # Resonance approx
            1.5,      0.8,      0.4,      0.15,     0.13,
            0.05,     0.01,     0.005
        ])

        # --- NEW: Inelastic Scattering [Barns] ---
        # U-238 Inelastic is CRITICAL. Threshold ~45 keV.
        self.sigma_inelastic_grid = np.array([
            0.0,      0.0,      0.0,      0.0,      0.0,
            0.0,      0.0,      0.0,      0.8,      2.5,
            2.8,      2.5,      2.4
        ])
        
        # Interpolators
        self.interpolator_total = interp1d(np.log(self.energy_grid), np.log(self.sigma_total_grid), kind='linear', fill_value="extrapolate")
        self.interpolator_fission = interp1d(np.log(self.energy_grid), np.log(self.sigma_fission_grid), kind='linear', fill_value="extrapolate")
        self.interpolator_capture = interp1d(np.log(self.energy_grid), np.log(self.sigma_capture_grid), kind='linear', fill_value="extrapolate")
        
        safe_inelastic = np.where(self.sigma_inelastic_grid == 0, 1e-20, self.sigma_inelastic_grid)
        self.interpolator_inelastic = interp1d(np.log(self.energy_grid), np.log(safe_inelastic), kind='linear', fill_value="extrapolate")

    def get_total_sigma(self, energy_ev):
        return np.exp(self.interpolator_total(np.log(energy_ev)))
    
    def get_sigma_fission(self, energy_ev):
        return np.exp(self.interpolator_fission(np.log(energy_ev)))

    def get_sigma_capture(self, energy_ev):
        return np.exp(self.interpolator_capture(np.log(energy_ev)))
    
    def get_sigma_inelastic(self, energy_ev):
        # Threshold for U238 is ~45 keV
        if energy_ev < 45000:
            return 0.0
        return np.exp(self.interpolator_inelastic(np.log(energy_ev)))

    def get_sigma_elastic(self, energy_ev):
        """
        Sigma_Elastic = Total - (Fission + Capture + Inelastic)
        """
        st = self.get_total_sigma(energy_ev)
        sf = self.get_sigma_fission(energy_ev)
        sc = self.get_sigma_capture(energy_ev)
        si = self.get_sigma_inelastic(energy_ev)
        return max(0.0, st - (sf + sc + si))