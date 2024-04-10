import torch
from transformers import LogitsProcessor


class GrammarLogitsProcessor(LogitsProcessor):
    """ Guide generation to match a grammar. """

    def __init__(self, grammar, main_grammar_rule, encode, decode, vocab_size, is_greedy, prefix_length, eos_token,
                 max_consider=None):
        self.grammar = grammar
        self.main_grammar_rule = main_grammar_rule
        self.decode = decode
        self.is_greedy = is_greedy
        self.prefix_length = prefix_length
        self.max_consider = max_consider if max_consider is not None else vocab_size
        self.bias_vector = torch.zeros(vocab_size)
        self.current_strings = None
        self.current_length = 0
        self.forced_chars = 0
        self.eos_token_id = encode(eos_token).input_ids[1]
        self.pattern_complete = False
        self.bias_value = 0

    def __call__(self, input_ids, scores):
        if self.pattern_complete:
            return scores
        self._extend_current_strings(input_ids)
        self._compute_bias_values(scores)
        return scores + self.bias_vector.to(scores.device)

    def _extend_current_strings(self, input_ids):
        if self.current_strings is None:
            self.current_strings = ""
        tokens = input_ids.clone().detach()[0, :]
        self.current_strings = self.decode(tokens)

        if self.current_length == 0:
            self.current_length = len(self.current_strings)
            self.current_strings = ""

        self.current_strings = self.current_strings[self.current_length:]

    def _compute_bias_values(self, scores):
        self.bias_vector[:] = 0
        scores_tensor = scores[0, :].detach().clone()
        sort_inds = torch.argsort(scores_tensor, 0, True)
        eos_out = False
        to_bias = self._find_matches_to_bias(sort_inds, scores_tensor)
        if len(to_bias) == 0:
            to_bias = [self.eos_token_id]
            self.pattern_complete = True
        else:
            eos_out = True
            # Corrected retrieval of the top score
        min_to_bias = float(scores[0, to_bias].min())
        self.bias_value = scores[0, sort_inds[0]] - min_to_bias + 1000  # make sure the tokens

        # Apply the bias to the tokens in to_bias
        self._apply_bias(to_bias, eos_out)

    def _find_matches_to_bias(self, sort_inds, scores_tensor):
        max_match_length = 0
        to_bias = []
        for i in range(min(sort_inds.shape[0], self.max_consider)):
            if self.pattern_complete:
                break
            cd = [sort_inds[i].item()]
            dec = self.decode(cd)
            proposed_string = (self.current_strings + dec)
            if len(dec) == 0:
                continue
            check = self._is_valid(proposed_string)

            if check:

                if len(proposed_string) > max_match_length:
                    max_match_length = len(proposed_string)
                    to_bias = cd
                    if self.is_greedy:
                        break
                    if self.pattern_complete:
                        break
        return to_bias

    def _is_valid(self, string):
        valid, only_partially = self.grammar.parse(string, self.main_grammar_rule)
        if valid:
            self.pattern_complete = not only_partially
            return True
        else:
            return False

    def _apply_bias(self, to_bias, eos_out):
        for x in to_bias:
            self.bias_vector[x] = self.bias_value
        if eos_out:
            self.bias_vector[self.eos_token_id] = -100000
