import numpy as np
from scipy.spatial.transform import Rotation as R

def matrix_to_sixd(matrix):
    """
    Converts a 3x3 rotation matrix to a 6D representation.
    We simply take the first two columns and flatten them.
    """
    # Extract first two columns (x and y axes)
    # shape (3, 3) -> (3, 2)
    two_cols = matrix[:, 0:2] 
    
    # Flatten in column-major order (Col 1 then Col 2) to match your previous format
    return two_cols.flatten(order='F')

def sixd_to_matrix(sixd_vector):
    """
    Reconstructs 3x3 matrix from 6D vector using Gram-Schmidt.
    """
    a1 = sixd_vector[0:3]
    a2 = sixd_vector[3:6]

    b1 = a1 / np.linalg.norm(a1)
    b3 = np.cross(b1, a2)
    b3 /= np.linalg.norm(b3)
    b2 = np.cross(b3, b1)

    return np.stack([b1, b2, b3], axis=1)

# --- EXECUTION ---

# 1. Start with a known Quaternion
# Example: 90 degree rotation around Z-axis
# Identity is [0, 0, 0, 1]. This is approx [0, 0, 0.707, 0.707]
initial_quat = np.array([0.0, 0.0, 1.0, 1.0]) 
initial_quat /= np.linalg.norm(initial_quat) # Normalize to be valid

# 2. Convert Quaternion -> Matrix
r_start = R.from_quat(initial_quat)
matrix_original = r_start.as_matrix()

# 3. Convert Matrix -> 6D
# This drops the 3rd column entirely.
sixd_rep = matrix_to_sixd(matrix_original)

# 4. Convert 6D -> Matrix (Reconstruction)
matrix_restored = sixd_to_matrix(sixd_rep)

# 5. Convert Matrix -> Quaternion
r_end = R.from_matrix(matrix_restored)
final_quat = r_end.as_quat()

# --- OUTPUT & VERIFICATION ---
print(f"1. Initial Quat:   {initial_quat}")
print(f"2. Original Matrix:\n{matrix_original}")
print(f"3. 6D Vector:      {sixd_rep}")
print(f"4. Restored Matrix:\n{matrix_restored}")
print(f"5. Final Quat:     {final_quat}")

# Check Difference (Account for double cover q == -q)
diff = np.min([
    np.linalg.norm(initial_quat - final_quat),
    np.linalg.norm(initial_quat + final_quat)
])
print(f"\nDifference: {diff:.8f}")