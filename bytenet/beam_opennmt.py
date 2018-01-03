from __future__ import division
import torch

"""
    Adapted from OpenNMT Beam Search (https://github.com/OpenNMT/OpenNMT-py).

"""

class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Notes: Removed "bos", beginning of sequence

    Args:
       size (int): beam size
       pad, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, eos,
                 n_best=1, cuda=False,
                 global_scorer=None):

        self.size = size
        self.scores = torch.FloatTensor(size).zero_()
        self.all_scores = []

        #self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        #self.scores = self.tt.FloatTensor(size).zero_()
        #self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.LongTensor(size)
                        .fill_(pad)]

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, char_probs):
        """
        Get the probablity of each character for every beam.

        K = beam size?

        #Given prob over words for every last beam `wordLk` and attention
        #`attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        #* `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """

        num_chars = char_probs.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = char_probs + \
                self.scores.unsqueeze(1).expand_as(char_probs)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = char_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_chars
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_chars))

        if self.global_scorer is not None:
            self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            # self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = 0. # beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1])
