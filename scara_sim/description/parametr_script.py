L1 = 0.4
L2 = 0.3

r = 0.025

ro = 4510

m1 = 3.14 * L1 * r**2 * ro
m2 = 3.14 * L2 * r**2 * ro

print(f'm1 = {m1}, m2 = {m2}')

Ixx1 = 1/2 * m1 * r**2

Iyy1 = Izz1 = m1/12 * (3*r**2 + L1**2)

Ixx2 = 1/2 * m2 * r**2

Iyy2 = Izz2 = m2/12 * (3*r**2 + L2**2)

print(f'Ixx1 = {Ixx1}, Iyy1 = Izz1 = {Iyy1}')
print(f'Ixx2 = {Ixx2}, Iyy2 = Izz2 = {Iyy2}')