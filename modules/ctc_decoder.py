"""
MMRT Module 8: CTC Beam Search Decoder
Sequence decoding for sign language gloss recognition.
"""
import torch
from itertools import groupby

try:
    import ctcdecode
    _HAS_CTCDECODE = True
except ImportError:
    _HAS_CTCDECODE = False


class CTCBeamSearchDecoder:
    """
    MMRT Module 8: CTC beam search decoder.
    Falls back to greedy decoding if ctcdecode is not available.
    """

    def __init__(self, gloss_dict, num_classes, search_mode='beam', blank_id=0, beam_width=10):
        self.i2g_dict = {v[0]: k for k, v in gloss_dict.items()}
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.ctc_decoder = None
        if _HAS_CTCDECODE and search_mode == 'beam':
            vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
            self.ctc_decoder = ctcdecode.CTCBeamDecoder(
                vocab, beam_width=beam_width, blank_id=blank_id, num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.ctc_decoder is not None and self.search_mode == 'beam':
            return self._beam_search(nn_output, vid_lgt, probs)
        return self._greedy_decode(nn_output, vid_lgt)

    def _beam_search(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(
            nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([self.i2g_dict[int(g)] for g in first_result
                             if int(g) in self.i2g_dict])
        return ret_list

    def _greedy_decode(self, nn_output, vid_lgt):
        """Greedy CTC decoding fallback."""
        index_list = torch.argmax(nn_output, dim=2)
        ret_list = []
        for batch_idx in range(len(index_list)):
            lgt = int(vid_lgt[batch_idx])
            seq = index_list[batch_idx][:lgt].tolist()
            # Collapse repeats and remove blank
            collapsed = [k for k, _ in groupby(seq) if k != self.blank_id]
            ret_list.append([self.i2g_dict[g] for g in collapsed if g in self.i2g_dict])
        return ret_list


# Alias for state_dict compatibility
Decode = CTCBeamSearchDecoder

