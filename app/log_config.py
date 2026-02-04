import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress Keras warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
