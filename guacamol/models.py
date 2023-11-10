import pickle

def load_pickled_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Preprocessor
MORGAN_PREPROCESSOR = load_pickled_model('models/morgan_preprocessor.pkl')

# Protein targets 
ACHE_MODEL = load_pickled_model('models/ache_model.pkl')
APP_MODEL = load_pickled_model('models/app_model.pkl')
D2R_MODEL = load_pickled_model('models/d2r_model.pkl')
_5HT1A_MODEL = load_pickled_model('models/5ht1a_model.pkl')
NTRK1_MODEL = load_pickled_model('models/nrtk1_model.pkl')
NTRK3_MODEL = load_pickled_model('models/nrtk3_model.pkl')
ROS1_MODEL = load_pickled_model('models/ros1_model.pkl')

# Other criterion
BBB_MODEL = load_pickled_model('models/bbb_model.pkl')