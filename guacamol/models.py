import pickle
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')

def load_pickled_model(model_path):
    if model_path is None:
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Preprocessor
MORGAN_PREPROCESSOR = load_pickled_model(os.path.join(MODEL_DIR, 'Morgan.pkl'))

# Protein targets 
ACHE_MODEL = load_pickled_model(None)
APP_MODEL = load_pickled_model(None)
D2R_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'D2R.pkl'))
_5HT1A_MODEL = load_pickled_model(os.path.join(MODEL_DIR, '_5HT1A.pkl'))
NTRK1_MODEL = load_pickled_model(None)
NTRK3_MODEL = load_pickled_model(None)
ROS1_MODEL = load_pickled_model(None)

# Other criterion
BBB_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'BBB.pkl'))