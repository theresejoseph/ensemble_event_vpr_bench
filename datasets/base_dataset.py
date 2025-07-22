from abc import ABC, abstractmethod
class BaseDataset(ABC):
    def __init__(self, base_path):
        self.base_path = base_path

        
    # @abstractmethod
    # def load_events(self, sequence_name):
    #     """Load a events from a specific sequence in the dataset"""
    #     pass
    
    @abstractmethod
    def process_sequence(self, sequence_name, reconstructor, force_reprocess=False):
        """Process sequence only if needed"""
        pass