from abc import ABC, abstractmethod
# reconstruction/base_reconstructor.py
class BaseReconstructor(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def reconstruct(self, events, time_windows):
        """Reconstruct frames from events"""
        pass
