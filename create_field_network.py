import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# -------------------------- 1. Basic Configuration --------------------------
# Core parameters: 128×128 spatial grid, 50 logtau layers (consistent with SIR requirements)
ny, nx, nlogtau = 128, 128, 50  # 128×128 Mm field-of-view (1 Mm/pixel), 50 log optical depth layers
model = np.zeros((ny, nx, nlogtau, 11), dtype=np.float32)  # SIR standard dimension: (ny, nx, nlogtau, 11)

# Fixed non-magnetic parameters (empirical values where not specified)
model[:, :, :, 0] = 5777.0    # 0: Temperature (K, typical photospheric value)
model[:, :, :, 1] = 0.0       # 1: LOS velocity (to be updated with random + downflow bias)
model[:, :, :, 3] = 1.0e5     # 3: Microturbulence (cm/s, recommended fixed value)
model[:, :, :, 4] = 0.0       # 4: Macroturbulence (km/s, default 0)
model[:, :, :, 7] = 0.0       # 7: Pressure (default 0, SIR-calculated)
model[:, :, :, 8] = 0.0       # 8: Electron density (default 0, SIR-calculated)
model[:, :, :, 9] = 1.0       # 9: Element abundance (default 1.0)
model[:, :, :, 10] = 0.0      # 10: Reserved for filling factor (f)

# -------------------------- 2. Core Generation Functions (Observation-based) --------------------------
def sample_power_law(alpha, x_min, x_max, n_samples):
    """Power-law sampling for magnetic flux (α=-1.85 from observational fits)"""
    u = np.random.uniform(0, 1, n_samples)
    exponent = alpha + 1
    return (u * (x_max**exponent - x_min**exponent) + x_min**exponent) ** (1/exponent)

def gaussian_flux_tube(x_grid, y_grid, x0, y0, r, B_peak):
    """Gaussian flux tube profile (models network elements with radial decay)"""
    dx = x_grid - x0
    dy = y_grid - y0
    return B_peak * np.exp(-(dx**2 + dy**2) / (2 * r**2))

def generate_small_bipoles(n_pairs, x_grid, y_grid, B_max=30, size=0.5):
    """Generate small bipoles (models internetwork short closed loops)"""
    bipole_field = np.zeros((ny, nx))
    for _ in range(n_pairs):
        # Random position
        x0 = np.random.uniform(0, nx-1)
        y0 = np.random.uniform(0, ny-1)
        # Random bipolar axis orientation
        angle = np.random.uniform(0, np.pi)
        # Positive/negative pole positions (separation = size)
        x_pos = x0 + size * np.cos(angle)
        y_pos = y0 + size * np.sin(angle)
        x_neg = x0 - size * np.cos(angle)
        y_neg = y0 - size * np.sin(angle)
        # Random strength (5-30 G, consistent with weak internetwork fields)
        B = np.random.uniform(5, B_max)
        # Superpose bipolar Gaussian profiles
        bipole_field += gaussian_flux_tube(x_grid, y_grid, x_pos, y_pos, size/2, B)
        bipole_field -= gaussian_flux_tube(x_grid, y_grid, x_neg, y_neg, size/2, B)
    return bipole_field

# -------------------------- 3. Grid Initialization --------------------------
x_coords = np.linspace(0, nx-1, nx)
y_coords = np.linspace(0, ny-1, ny)
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

# -------------------------- 4. Network Magnetic Field Generation (strong, clustered) --------------------------
# 4.1 Network parameters
n_network = 512  # Number of network elements (scaled from 200 for 50×50 grid: ~6.55× area)
alpha_flux = -1.85  # Flux power-law exponent (observational fit)
phi_min, phi_max = 1e17, 1e19  # Network flux range (Mx, recommended interval)
B_peak_mu, B_peak_sigma = 1200, 300  # Peak field normal distribution (G)
polarity_bias = 0.85  # Coronal hole unipolar bias (85% same polarity)

# 4.2 Sample network parameters
phi_network = sample_power_law(alpha_flux, phi_min, phi_max, n_network)  # Power-law flux sampling
B_peak_network = np.random.normal(B_peak_mu, B_peak_sigma, n_network)  # Peak field sampling
B_peak_network = np.clip(B_peak_network, 500, 1500)  # Constrain to 500-1500 G
r_network = np.sqrt(phi_network / (np.pi * B_peak_network)) / 1e3  # Radius (Mm, 150-600 km)
r_network = np.clip(r_network, 0.15, 0.6)  # Constrain radius to 0.15-0.6 Mm
network_polarities = np.where(np.random.rand(n_network) < polarity_bias, 1, -1)  # Polarity bias

# 4.3 Initialize network magnetic grids
B_network = np.zeros((ny, nx))
gamma_network = np.full((ny, nx), np.deg2rad(45))  # Inclination default
chi_network = np.zeros((ny, nx))  # Azimuth default

# 4.4 Place network flux tubes
for i in range(n_network):
    x0 = np.random.uniform(0, nx-1)  # Random position
    y0 = np.random.uniform(0, ny-1)
    # Generate Gaussian flux tube
    tube = gaussian_flux_tube(x_grid, y_grid, x0, y0, r_network[i], B_peak_network[i] * network_polarities[i])
    # Superpose tubes (max to avoid cancellation)
    B_network = np.maximum(B_network, np.abs(tube)) * np.sign(tube + B_network)
    # Network inclination (vertical bias: Normal(15°,10°))
    mask = tube != 0
    gamma_val = np.clip(np.random.normal(15, 10), 0, 90)  # Constrain to 0-90°
    gamma_network[mask] = np.deg2rad(gamma_val)
    # Network azimuth (uniform random 0-360°)
    chi_network[mask] = np.deg2rad(np.random.uniform(0, 360))

# -------------------------- 5. Internetwork Magnetic Field Generation (weak, diffuse) --------------------------
# 5.1 Internetwork parameters
n_bipoles = 4096  # Number of bipoles (scaled from 1000 for 50×50 grid)
internetwork_B_rms = 40  # RMS field strength (30-70 G typical)
correlation_scale = 0.2  # Spatial correlation scale (0.2 Mm = 200 km)

# 5.2 Generate smooth noise background (small-scale correlation)
noise = np.random.normal(0, internetwork_B_rms, (ny, nx))
internetwork_bg = gaussian_filter(noise, sigma=correlation_scale)  # Gaussian smoothing

# 5.3 Superpose small bipoles (models short loops)
internetwork_bipoles = generate_small_bipoles(n_bipoles, x_grid, y_grid, B_max=30, size=0.5)

# 5.4 Total internetwork field
B_internetwork = internetwork_bg + internetwork_bipoles

# 5.5 Internetwork inclination (mixed: 60% horizontal-dominated + 40% isotropic)
gamma_internetwork = np.zeros((ny, nx))
mask_horizontal = np.random.rand(ny, nx) < 0.6
gamma_horizontal = np.clip(np.random.normal(80, 15, size=mask_horizontal.sum()), 0, 90)
gamma_internetwork[mask_horizontal] = np.deg2rad(gamma_horizontal)
gamma_internetwork[~mask_horizontal] = np.deg2rad(np.random.uniform(0, 90, size=(~mask_horizontal).sum()))

# 5.6 Internetwork azimuth (uniform random)
chi_internetwork = np.deg2rad(np.random.uniform(0, 360, (ny, nx)))

# -------------------------- 6. Merge Network and Internetwork Fields --------------------------
network_mask = np.abs(B_network) > 50  # Network mask (B > 50 G threshold)
B_total = np.where(network_mask, B_network, B_internetwork)  # Merged field strength
gamma_total = np.where(network_mask, gamma_network, gamma_internetwork)  # Merged inclination
chi_total = np.where(network_mask, chi_network, chi_internetwork)  # Merged azimuth

# -------------------------- 7. Filling Factor Calculation (empirical formula) --------------------------
filling_factor = np.clip(np.abs(B_total) / 1500, 0.01, 1.0)
filling_factor += np.random.normal(0, 0.02, size=filling_factor.shape)  # Add small random scatter
filling_factor = np.clip(filling_factor, 0.01, 1.0)  # Re-constrain bounds

# -------------------------- 8. LOS Velocity Configuration --------------------------
v_los = np.random.normal(0, 0.5, (ny, nx))  # Background random (σ=0.5 km/s)
v_los[network_mask] -= 0.3  # Downflow bias in strong field regions

# -------------------------- 9. Populate SIR Model Array --------------------------
model[:, :, :, 1] = v_los[:, :, np.newaxis]  # LOS velocity (constant across logtau)
model[:, :, :, 2] = np.abs(B_total)[:, :, np.newaxis]  # Magnetic field strength
model[:, :, :, 5] = gamma_total[:, :, np.newaxis]  # Inclination (radians)
model[:, :, :, 6] = chi_total[:, :, np.newaxis]  # Azimuth (radians)
model[:, :, :, 10] = filling_factor[:, :, np.newaxis]  # Filling factor

# -------------------------- 10. Save SIR Input File --------------------------
np.save('network_field.npy', model)
print("SIR input file generated: network_field.npy")
print(f"Dimensions: (ny={ny}, nx={nx}, nlogtau={nlogtau}, 11)")
print(f"Field strength range: {np.min(np.abs(B_total)):.1f}–{np.max(np.abs(B_total)):.1f} G")
print(f"Network area fraction: {np.mean(network_mask):.1%}")
print(f"Filling factor range: {np.min(filling_factor):.3f}–{np.max(filling_factor):.3f}")

# -------------------------- 11. Magnetic Field Visualization --------------------------
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)
cmap_mag = plt.cm.jet.copy()
cmap_comp = LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'white', 'red'], 256)

fig, axes = plt.subplots(3, 3, constrained_layout=True)
fig.suptitle('Network/Internetwork Magnetic Field Distribution', fontsize=14)

# 11.1 Total field strength
vmin_mag = np.percentile(np.abs(B_total), 1)
vmax_mag = np.percentile(np.abs(B_total), 99)
im1 = axes[0,0].imshow(np.abs(B_total), cmap='plasma', vmin=vmin_mag, vmax=vmax_mag)
axes[0,0].set_title('Field Strength (G)')
axes[0,0].set_xlabel('X Grid (Mm)')
axes[0,0].set_ylabel('Y Grid (Mm)')
plt.colorbar(im1, ax=axes[0,0], shrink=0.8)

# 11.2 Inclination (degrees)
gamma_deg = np.rad2deg(gamma_total)
im2 = axes[0,1].imshow(gamma_deg, cmap='jet', vmin=0, vmax=90)
axes[0,1].set_title('Inclination (deg.)')
axes[0,1].set_xlabel('X Grid (Mm)')
axes[0,1].set_ylabel('Y Grid (Mm)')
plt.colorbar(im2, ax=axes[0,1], shrink=0.8)

# 11.3 Azimuth (degrees)
chi_deg = np.rad2deg(chi_total)
im3 = axes[0,2].imshow(chi_deg, cmap='jet', vmin=0, vmax=360)
axes[0,2].set_title('Azimuth (deg.)')
axes[0,2].set_xlabel('X Grid (Mm)')
axes[0,2].set_ylabel('Y Grid (Mm)')
plt.colorbar(im3, ax=axes[0,2], shrink=0.8)

Bx = B_total * np.sin(gamma_total) * np.cos(chi_total)
By = B_total * np.sin(gamma_total) * np.sin(chi_total)
Bz = B_total * np.cos(gamma_total)

im4 = axes[1,0].imshow(Bx, cmap='bwr', vmin=-vmax_mag, vmax=vmax_mag)
axes[1,0].set_title('Bx (G)')
axes[1,0].set_xlabel('X Grid (Mm)')
axes[1,0].set_ylabel('Y Grid (Mm)')
plt.colorbar(im4, ax=axes[1,0], shrink=0.8)

im5 = axes[1,1].imshow(By, cmap='bwr', vmin=-vmax_mag, vmax=vmax_mag)
axes[1,1].set_title('By (G)')
axes[1,1].set_xlabel('X Grid (Mm)')
axes[1,1].set_ylabel('Y Grid (Mm)')
plt.colorbar(im5, ax=axes[1,1], shrink=0.8)

im6 = axes[1,2].imshow(Bz, cmap='bwr', vmin=-vmax_mag, vmax=vmax_mag)
axes[1,2].set_title('Bz (G)')
axes[1,2].set_xlabel('X Grid (Mm)')
axes[1,2].set_ylabel('Y Grid (Mm)')
plt.colorbar(im6, ax=axes[1,2], shrink=0.8)

# 11.4 Filling factor
im7 = axes[2,0].imshow(filling_factor, cmap='jet', vmin=0.01, vmax=1.0)
axes[2,0].set_title('Filling Factor')
axes[2,0].set_xlabel('X Grid (Mm)')
axes[2,0].set_ylabel('Y Grid (Mm)')
plt.colorbar(im7, ax=axes[2,0], shrink=0.8)

# 11.5 LOS velocity
im8 = axes[2,1].imshow(v_los, cmap='bwr', vmin=-2, vmax=2)
axes[2,1].set_title('LOS Velocity (km/s)')
axes[2,1].set_xlabel('X Grid (Mm)')
axes[2,1].set_ylabel('Y Grid (Mm)')
plt.colorbar(im8, ax=axes[2,1], shrink=0.8)

# 11.6 Network/internetwork mask
im9 = axes[2,2].imshow(network_mask, cmap='binary')
axes[2,2].set_title('Network (white)/Internetwork (black)')
axes[2,2].set_xlabel('X Grid (Mm)')
axes[2,2].set_ylabel('Y Grid (Mm)')
plt.colorbar(im9, ax=axes[2,2], shrink=0.8, ticks=[0,1])

# Save visualization
plt.savefig('network_field.npy.png', dpi=300, bbox_inches='tight')
print("field visualization saved: network_field.npy.png")
plt.show()