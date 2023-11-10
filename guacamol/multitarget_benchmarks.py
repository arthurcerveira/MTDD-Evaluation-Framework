from rdkit import Chem

from guacamol.common_scoring_functions import TargetResponseScoringFunction
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
# from guacamol.score_modifier import MinGaussianModifier, MaxGaussianModifier, ClippedScoreModifier, GaussianModifier
from guacamol.scoring_function import GeometricMeanScoringFunction
# from guacamol.utils.descriptors import num_rotatable_bonds, num_aromatic_rings, logP, qed, tpsa, bertz, mol_weight, \
    # AtomCounter, num_rings

from guacamol.models import (
    MORGAN_PREPROCESSOR,
    ACHE_MODEL,
    APP_MODEL,
    D2R_MODEL,
    _5HT1A_MODEL,
    NTRK1_MODEL,
    NTRK3_MODEL,
    ROS1_MODEL,
    BBB_MODEL
)


def alzheimer_mpo_benchmark() -> GoalDirectedBenchmark:
    """
    Benchmark to evaluate multi-target molecules active against Alzheimer's disease.
    Targets considered:
    - Acetylcholinesterase (AChE)
    - Amyloid-β peptide (APP)
    Other criteria:
    - Pass through blood-brain barrier (BBB)
    """
    ache_scorer = TargetResponseScoringFunction(
        target='AChE', model=ACHE_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    app_scorer = TargetResponseScoringFunction(
        target='APP', model=APP_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    bbb_scorer = TargetResponseScoringFunction(
        target='BBB', model=BBB_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    mean_scorer = GeometricMeanScoringFunction(
        [ache_scorer, app_scorer, bbb_scorer]
    )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name='Alzheimer MPO',
        objective=mean_scorer,
        contribution_specification=specification
    )


def schizophrenia_mpo_benchmark() -> GoalDirectedBenchmark:
    """
    Benchmark to evaluate multi-target molecules active against schizophrenia.
    Targets considered:
    - Dopamine D2 receptor (D2)
    - 5-hydroxytryptamine receptor 2A (5-HT2A)
    Other criteria:
    - Pass through blood-brain barrier (BBB)
    """
    d2_scorer = TargetResponseScoringFunction(
        target='D2', model=D2R_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    _5ht2a_scorer = TargetResponseScoringFunction(
        target='5-HT2A', model=_5HT1A_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    bbb_scorer = TargetResponseScoringFunction(
        target='BBB', model=BBB_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    mean_scorer = GeometricMeanScoringFunction(
        [d2_scorer, _5ht2a_scorer, bbb_scorer]
    )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name='Schizophrenia MPO',
        objective=mean_scorer,
        contribution_specification=specification
    )


def lung_cancer_mpo_benchmark() -> GoalDirectedBenchmark:
    """
    Benchmark to evaluate multi-target molecules active against lung cancer.
    Targets considered:
    - NT-3 growth factor receptor (NTRK3)
    - High affinity nerve growth factor receptor (NTRK1)
    - Proto-oncogene tyrosine-protein kinase ROS (ROS1)
    """
    ntrk3_scorer = TargetResponseScoringFunction(
        target='NTRK3', model=NTRK3_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    ntrk1_scorer = TargetResponseScoringFunction(
        target='NTRK1', model=NTRK1_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    ros1_scorer = TargetResponseScoringFunction(
        target='ROS1', model=ROS1_MODEL, preprocess_smiles=MORGAN_PREPROCESSOR
    )

    mean_scorer = GeometricMeanScoringFunction(
        [ntrk3_scorer, ntrk1_scorer, ros1_scorer]
    )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name='Lung cancer MPO',
        objective=mean_scorer,
        contribution_specification=specification
    )
