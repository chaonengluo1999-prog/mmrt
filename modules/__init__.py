# MMRT Modules Package

# Inference-critical modules (always available)
from .resnet_encoder import build_visual_backbone, MMRTVisualBackbone
from .temporal_conv import MultiScaleTemporalEncoder, TemporalConv
from .bilstm_temporal import TemporalSequenceModel, BiLSTMLayer
from .ctc_decoder import CTCBeamSearchDecoder, Decode
from .criterions import SequenceKnowledgeDistillation, SeqKD
from .video_transforms import extract_video_frames
from .llm_postprocessor import GlossTranslator

# Optional heavy modules (require transformers, mediapipe, etc.)
def _lazy_import(name):
    try:
        import importlib
        return importlib.import_module(f'.{name}', package=__name__)
    except ImportError:
        return None

__all__ = [
    'build_visual_backbone', 'MMRTVisualBackbone',
    'MultiScaleTemporalEncoder', 'TemporalConv',
    'TemporalSequenceModel', 'BiLSTMLayer',
    'CTCBeamSearchDecoder', 'Decode',
    'SequenceKnowledgeDistillation', 'SeqKD',
    'extract_video_frames',
    'GlossTranslator',
]
