from typing import Callable, List

from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import pandas as pd

from guacamol.utils.descriptors import mol_weight, logP, num_H_donors, tpsa, num_atoms, AtomCounter
from guacamol.utils.fingerprints import get_fingerprint
from guacamol.score_modifier import ScoreModifier, MinGaussianModifier, MaxGaussianModifier, GaussianModifier
from guacamol.scoring_function import ScoringFunctionBasedOnRdkitMol, MoleculewiseScoringFunction, BatchScoringFunction
from guacamol.utils.chemistry import smiles_to_rdkit_mol, parse_molecular_formula
from guacamol.utils.math import arithmetic_mean, geometric_mean


class TargetResponseScoringFunction(BatchScoringFunction):
    """
    Scoring function that measures the response of a molecule against a target protein.
    - Requires a pre-trained QSAR model with a predict method
    - Requires a preprocessor method that converts a SMILES string (e.g. Morgan fingerprint)
    """

    def __init__(self, target, model, preprocessor, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            target: target protein
            model: pre-trained QSAR model
            preprocess_method: method to preprocess a SMILES string
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)

        self.target = target
        self.model = model
        self.preprocessor = preprocessor

    def raw_score_list(self, smiles_list: List[str]) -> List[float]:
        mols = [smiles_to_rdkit_mol(smiles) for smiles in smiles_list]

        predicted_activity_proba = self.model_predict_wrapper(mols)

        return predicted_activity_proba

    def model_predict_wrapper(self, mols: List[Chem.Mol]):
        mol_features_computed = True

        try:
            mol_features = self.preprocessor.vectorized_compute_features(mols)
        except:
            mol_features_computed = False

        predicted_activity_proba = None

        if mol_features_computed:
            df_features = pd.DataFrame(mol_features, columns=self.preprocessor.features)

            predicted_activity_proba = self.model.predict_proba(df_features)
            predicted_activity_proba = predicted_activity_proba[:,1]

        return predicted_activity_proba


class SyntheticAccessibilityScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Scoring function that measures the synthetic accessibility of a molecule.
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00678-z
    Accessibility scorers:
    - SAscore
    - SYBA
    - SCScore
    - RAscore
    This function expects the accessibility_scorer function to
    return Lower scores as higher synthetic accessibility.
    """

    def __init__(self, accessibility_scorer, min_score=1, max_score=10, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)
        self.accessibility_scorer = accessibility_scorer
        self.min_score = min_score
        self.max_score = max_score

    def score_mol(self, mol: Chem.Mol) -> float:
        accessibility = self.accessibility_scorer(mol)

        # Normalize accessibility score to [0,1]
        normalized_accessibility = (accessibility - self.min_score) / (self.max_score - self.min_score)
        # Invert accessibility score to [1,0] (higher score means higher synthetic accessibility)
        accessibility_score = 1 - normalized_accessibility

        return accessibility_score


class RdkitScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Scoring function wrapping RDKit descriptors.
    """

    def __init__(self, descriptor: Callable[[Chem.Mol], float], score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            descriptor: molecular descriptors, such as the ones in descriptors.py
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)
        self.descriptor = descriptor

    def score_mol(self, mol: Chem.Mol) -> float:
        return self.descriptor(mol)


class TanimotoScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Scoring function that looks at the fingerprint similarity against a target molecule.
    """

    def __init__(self, target, fp_type, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            target: target molecule
            fp_type: fingerprint type
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)

        self.target = target
        self.fp_type = fp_type
        target_mol = smiles_to_rdkit_mol(target)
        if target_mol is None:
            raise RuntimeError(f'The similarity target {target} is not a valid molecule.')

        self.ref_fp = get_fingerprint(target_mol, self.fp_type)

    def score_mol(self, mol: Chem.Mol) -> float:
        fp = get_fingerprint(mol, self.fp_type)
        return TanimotoSimilarity(fp, self.ref_fp)

class RuleOfFiveScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Lipinski's rule of five scoring function describes the drug-likeness of a chemical compound
    This scoring function considers the following criteria:
    - no more than 5 hydrogen bond donors
    - no more than 10 hydrogen bond acceptors
    - a molecular weight under 500 daltons
    - an octanol-water partition coefficient log P not greater than 5
    """

    def __init__(self, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)

    def score_mol(self, mol: Chem.Mol) -> float:
        # Calculate the number of violations of the rule of five
        violations = 0

        if Lipinski.NumHDonors(mol) > 5:
            violations += 1
        if Lipinski.NumHAcceptors(mol) > 10:
            violations += 1
        if mol_weight(mol) > 500:
            violations += 1
        if logP(mol) > 5:
            violations += 1

        # An orally active drug-like compound shouldn't have more than one violation of the RO5
        if violations <= 1:  # 0 or 1 violations
            return 1.0

        ro5_score = 1 - violations / 4

        return ro5_score


class CNS_MPO_ScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    CNS MPO scoring function
    https://pubs.acs.org/doi/10.1021/jm501535r
    """

    def __init__(self, max_logP=5.0, maxMW=360, min_tpsa=40, max_tpsa=90, max_hbd=0) -> None:
        super().__init__()

        self.logP_gauss = MinGaussianModifier(max_logP, 1)
        self.molW_gauss = MinGaussianModifier(maxMW, 60)
        self.tpsa_maxgauss = MaxGaussianModifier(min_tpsa, 20)
        self.tpsa_mingauss = MinGaussianModifier(max_tpsa, 30)
        self.hbd_gauss = MinGaussianModifier(max_hbd, 2.0)

    def score_mol(self, mol: Chem.Mol) -> float:
        mw = mol_weight(mol)
        lp = logP(mol)
        hbd = num_H_donors(mol)
        mol_tpsa = tpsa(mol)

        o1 = self.tpsa_mingauss(mol_tpsa)
        o2 = self.tpsa_maxgauss(mol_tpsa)
        o3 = self.hbd_gauss(hbd)
        o4 = self.logP_gauss(lp)
        o5 = self.molW_gauss(mw)

        return 0.2 * (o1 + o2 + o3 + o4 + o5)


class IsomerScoringFunction(MoleculewiseScoringFunction):
    """
    Scoring function for closeness to a molecular formula.

    The score penalizes deviations from the required number of atoms for each element type, and for the total
    number of atoms.

    F.i., if the target formula is C2H4, the scoring function is the average of three contributions:
    - number of C atoms with a Gaussian modifier with mu=2, sigma=1
    - number of H atoms with a Gaussian modifier with mu=4, sigma=1
    - total number of atoms with a Gaussian modifier with mu=6, sigma=2
    """

    def __init__(self, molecular_formula: str, mean_function='geometric') -> None:
        """
        Args:
            molecular_formula: target molecular formula
            mean_function: which function to use for averaging: 'arithmetic' or 'geometric'
        """
        super().__init__()

        self.mean_function = self.determine_mean_function(mean_function)
        self.scoring_functions = self.determine_scoring_functions(molecular_formula)

    @staticmethod
    def determine_mean_function(mean_function: str) -> Callable[[List[float]], float]:
        if mean_function == 'arithmetic':
            return arithmetic_mean
        if mean_function == 'geometric':
            return geometric_mean
        raise ValueError(f'Invalid mean function: "{mean_function}"')

    @staticmethod
    def determine_scoring_functions(molecular_formula: str) -> List[RdkitScoringFunction]:
        element_occurrences = parse_molecular_formula(molecular_formula)

        total_number_atoms = sum(element_tuple[1] for element_tuple in element_occurrences)

        # scoring functions for each element
        functions = [RdkitScoringFunction(descriptor=AtomCounter(element),
                                          score_modifier=GaussianModifier(mu=n_atoms, sigma=1.0))
                     for element, n_atoms in element_occurrences]

        # scoring functions for the total number of atoms
        functions.append(RdkitScoringFunction(descriptor=num_atoms,
                                              score_modifier=GaussianModifier(mu=total_number_atoms, sigma=2.0)))

        return functions

    def raw_score(self, smiles: str) -> float:
        # return the average of all scoring functions
        scores = [f.score(smiles) for f in self.scoring_functions]
        if self.corrupt_score in scores:
            return self.corrupt_score
        return self.mean_function(scores)


class SMARTSScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Tests for SMARTS which should be or should not be present in the compound.


    """

    def __init__(self, target: str, inverse=False) -> None:
        """

        :param target: The SMARTS string to match.
        :param inverse: Specifies whether the SMARTS is desired (False) or an antipattern, which we don't want to see
                        in the molecules (inverse=False)
        """
        super().__init__()
        self.inverse = inverse
        self.smarts = target
        self.target = Chem.MolFromSmarts(target)

        assert target is not None

    def score_mol(self, mol: Chem.Mol) -> float:

        matches = mol.GetSubstructMatches(self.target)

        if len(matches) > 0:
            if self.inverse:
                return 0.0
            else:
                return 1.0
        else:
            if self.inverse:
                return 1.0
            else:
                return 0.0
