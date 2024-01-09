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
ACHE_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'AChE.pkl'))
APP_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'APP.pkl'))
D2R_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'D2R.pkl'))
D3R_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'D3R.pkl'))
D4R_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'D4R.pkl'))
_5HT1A_MODEL = load_pickled_model(os.path.join(MODEL_DIR, '_5HT1A.pkl'))
NTRK1_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'NTRK1.pkl'))
NTRK3_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'NTRK3.pkl'))
ROS1_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'ROS1.pkl'))

# Other criterion
BBB_MODEL = load_pickled_model(os.path.join(MODEL_DIR, 'BBB.pkl'))