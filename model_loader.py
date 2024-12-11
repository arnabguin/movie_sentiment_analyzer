#!python

from lib.constants import model_predicates
from lib.constants import model_parms
from keras import models

class ModelLoader:

    def __init__(self, model_name):    
        model_predicates.authorize()
        self.loaded_model = models.load_model(model_name + model_parms.FILE_EXT_DELIMITER + model_parms.MODEL_PERSISTENT_FORMAT)

    def get_loaded_model(self):
        return self.loaded_model
