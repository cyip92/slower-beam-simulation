import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

# Slowing parameters
path_length = 0.443
linewidth = 32.5e6
freq = 2 * 365.1582e12
detuning = -220e6
norm_int = 10    # Normalized intensity, I/I_sat

# Atomic parameters and constants (all units are SI)
m = 164.93 * 1.661e-27
boltzmann = 1.381e-23
atom_temperature = 1130 + 273.15
c = 299792458
h = 6.626e-34


# Generating a random thermal velocity vector and projecting it onto one axis is the same as just drawing from three
# independent 1D Gaussian distributions and taking the magnitude.  The three samples ensure the density of states
# scales as v^2 just like the actual Maxwell-Boltzmann speed distribution does.
def get_random_velocity(temperature):
    s = math.sqrt(boltzmann * temperature / m)
    x = abs(np.random.normal(0, s))
    y = abs(np.random.normal(0, s))
    z = abs(np.random.normal(0, s))
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


# Spontaneous decay happens uniformly in any direction and does affect the momentum of the atom, but needs to be
# generated carefully to ensure isotropy.  Returns the projection of a random point on the unit sphere onto an axis,
# Algorithm taken from https://mathworld.wolfram.com/SpherePointPicking.html and extraneous steps were removed.
def get_random_emission():
    s = 0
    while True:
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        s = x1 ** 2 + x2 ** 2
        if s < 1:
            return 1 - 2 * s


vel_array_unslowed = []
vel_array_slowed = []
trials = 100
for n in tqdm(range(trials), ascii=True):
    vel = get_random_velocity(atom_temperature)
    vel_array_unslowed.append(vel)

    distance = 0
    while 0 <= distance < path_length:
        # Binomial approximation of Doppler shift for performance, since v << c
        doppler_detuning = detuning + freq * vel / (2 * c)
        scattering_rate = linewidth / 2 * norm_int / (1 + norm_int + (2 * doppler_detuning / linewidth) ** 2)
        distance += vel * (1 / scattering_rate + 1 / linewidth)
        vel -= h * freq / (m * c) * (1 - get_random_emission())
    vel_array_slowed.append(vel)

plt.hist(vel_array_unslowed, alpha=0.5, bins=100, label='Unslowed')
plt.hist(vel_array_slowed, alpha=0.5, bins=100, label='Slowed')
plt.legend(loc='upper right')
plt.ylabel('Occurrences')
plt.xlabel('Velocity (m/s)')
plt.show()
