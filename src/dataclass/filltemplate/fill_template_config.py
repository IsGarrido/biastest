from dataclasses import dataclass

@dataclass(frozen=True)
class  FillTemplateConfig(object):

    label: str
    """Readable experiment label"""

    templates_path: str
    """Templates path"""

    models_path: str
    """Models path"""

    n_predictions: int = 29
    """Number of predictions"""

    pos_tag_wanted: str = 'AQ'
    """Keep only words with this POS tag"""

    predict_sentiment: bool = False

    @property
    def n_dimensions(self): 
        return len(self.dimensions)

    @property
    def predict_pos_tag(self):
        return self.pos_tag_wanted != ''