#
# print all constants and corresponding values in the compact files
# The user can decide which subdetectors or other elements to show or not
#
import ROOT
import dd4hep
import os
import sys
import argparse
from dd4hep import Position, tesla, mm
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
        description="Print selected constants from a DD4hep compact file. Enable/disable the subdetectors in the elementsToShow list"
    )
parser.parse_args()

   

# ------------------------------------------------------------------
# Load detector geometry
# ------------------------------------------------------------------
compactFile = "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml"
# compactFile = "ILD/compact/ILD_s6_v02/ILD_s6_v02.xml"
path_to_detector = os.environ.get("K4GEO", "")
detectorFile = path_to_detector + "/" + compactFile
detector = dd4hep.Detector.getInstance()
detector.fromXML(detectorFile)
print("Loaded detector from compact file:", detectorFile)
        

# Grid definition
# rho_vals = np.linspace(2650, 2750, 100) * mm   # 0 - 6 m
rho_vals = np.linspace(0, 6000, 100) * mm   # 0 - 6 m
z_vals   = np.linspace(-5000, 5000, 100) * mm  # -5 - 5 m
# z_vals   = np.linspace(3060, 3070, 100) * mm  # -5 - 5 m

R, Z = np.meshgrid(rho_vals, z_vals)

# Arrays for field
Brho = np.zeros_like(R, dtype=float)
Bz   = np.zeros_like(Z, dtype=float)
Bmag = np.zeros_like(R, dtype=float)

# Compute field
print("rho (mm)     z (mm)  Bx (T)     Bx (T)     Bz (T)")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        rho = R[i, j]
        z   = Z[i, j]
        
        # phi = 0 ? x=rho, y=0
        pos = Position(rho, 0.0, z)
        B = detector.field().magneticField(pos) / tesla
        
        # cylindrical components (phi = 0)
        Br = B.x()
        Bz_val = B.z()
        
        Brho[i, j] = Br
        Bz[i, j]   = Bz_val
        Bmag[i, j] = np.sqrt(Br**2 + B.y()**2 + Bz_val**2)

        print("{:12f} {:12f} {:12f} {:12f} {:12f}".format(rho/mm, z/mm, B.x(), B.y(), B.z()))

# Convert axes to mm for plotting
R_mm = R / mm
Z_mm = Z / mm
        
# --- Plot ---
plt.figure(figsize=(8, 6))

# Background: |B|
plt.pcolormesh(Z_mm, R_mm, Bmag, shading='auto')
plt.colorbar(label='|B| [T]')

# Quiver: downsample for readability
step = 2
plt.quiver(
    Z_mm[::step, ::step],
    R_mm[::step, ::step],
    Bz[::step, ::step],
    Brho[::step, ::step],
    color='black',
    scale=100,
    headlength=2,
#    headwidth=3,
    headaxislength=1,
    width=0.003
)

plt.xlabel('z [mm]')
plt.ylabel('r [mm]')
plt.title('Magnetic Field Map (r-z plane)')
plt.tight_layout()

plt.savefig("field_map.pdf", format="pdf", bbox_inches="tight")
plt.show()
