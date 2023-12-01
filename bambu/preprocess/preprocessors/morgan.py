from rdkit.Chem import AllChem
import numpy as np

class MorganPreprocessor:

    def __init__(self, bits, radius):
        self.bits = bits
        self.radius = radius
        self.features = [f'morgan_fp_bit_{i}' for i in range(self.bits)]
        self.vectorized_compute_features = np.vectorize(
            self.compute_features, signature='()->(n)'
        )

    def compute_features(self,mol):
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=self.radius, nBits = self.bits, bitInfo={})
        fingerprints = np.array(fingerprints)

        return fingerprints
