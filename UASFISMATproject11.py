import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# UASFISMAT Project: Nomor 11 Tensor, Hitung tensor inersia dan tentukan nilai serta arah sumbu rotasi alami. 
# Definisi massa dan posisi

mass      = np.array([1, 2, 1])
positions = np.array([
    [0, 1, 1], 
    [1, 0, 1], 
    [1, 1, 0]
])

# Tensor Inersia I
I = np.zeros ((3, 3))

for i in range(len(mass)):
    r         = positions[i]
    m         = mass[i]
    r_squared = np.dot(r, r)
    outer_rr  = np.outer(r, r)
    I += m * (r_squared * np.eye(3) - outer_rr)

print("Tensor Inersia I:\n", I)
print(I)

eigenvalues, eigenvectors = np.linalg.eig(I)
print("\nNilai Eigen (momen inersia):", eigenvalues)
print("Vektor Eigen (sumbu rotasi):\n", eigenvectors)

# UASFISMAT Project: Nomor 13 Kalkulus Variasi,Tentukan lintasan ϕ(r) yang meminimalkan jarak yang dilalui air di permukaan kerucut  
# Ukuran sudut kerucut
cot_alpha       = 1.0
ArithmeticError = 1 + cot_alpha**2

def geodesic_eq(r, Y):
    phi, dphi_dr = Y
    d2phi_dr2    = - (-1 / r) * dphi_dr
    return np.vstack((dphi_dr, d2phi_dr2))
def bc(ya, yb):
    return np.array([ya[0], yb[0] - np.pi / 2])

r              = np.linspace (1.0, 5.0, 100)
phi_guess      = np.linspace(0, np.pi / 2, r.size)
dphi_guess     = np.gradient(phi_guess, r)
Y_guess        = np.vstack((phi_guess, dphi_guess))

# Penyelesaian BVP
solve_bvp_result = solve_bvp(geodesic_eq, bc, r, Y_guess)
phi_analytic     = np.arccos(np.clip(1 / r, -1, 1))

if solve_bvp_result.success:
    plt.plot(solve_bvp_result.x, solve_bvp_result.y[0], label='Solusi Numerik ϕ(r)', color='blue')
    print("Penyelesaian BVP berhasil:", solve_bvp_result.message)
else:
    print("Penyelesaian BVP gagal:", solve_bvp_result.message)

# Plot hasil
plt.plot(r, phi_analytic, '--', label='Solusi Analitik ϕ(r) = arccos(1/r)', color='orange')
plt.title('Lintasan ϕ(r) Minimum di Permukaan Kerucut')
plt.xlabel('r')
plt.ylabel('ϕ(r)')
plt.grid()
plt.legend()
plt.savefig("geodesik_kerucut.png", dpi=300)
plt.show()