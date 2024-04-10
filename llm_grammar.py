import regex
from functools import lru_cache


class LLMGrammar:
    def __init__(self):
        self.rules = {}
        self.memo = {}
        self.verbose = False
        self.last_checked_string = ""
        self.last_checked_rule_name = ""

    def add_rule(self, rule):
        self.rules[rule.element_name] = rule

    def parse(self, string, rule_name, verbose=False):
        if (not string.startswith(self.last_checked_string)) or rule_name != self.last_checked_rule_name:
            self.memo = {}
        self.verbose = verbose
        success, end_position, parsed_elements, error, matched_only_partially = self.parse_rule(self.rules[rule_name],
                                                                                                string, 0)
        self.last_checked_string = string
        self.last_checked_rule_name = rule_name
        if verbose:
            if success and end_position == len(string) and error is None:
                return True, matched_only_partially, parsed_elements
            else:
                return False, False, f"Parsing error at position {end_position}: {error}"
        else:
            if success and end_position == len(string) and error is None:
                return True, matched_only_partially
            else:
                return False, False

    @lru_cache(maxsize=500000)
    def parse_rule(self, rule, string, position):
        if (rule.element_name, position) in self.memo:
            return self.memo[(rule.element_name, position)]
        if not rule:
            return False, position, [], f"Rule '{rule.element_name}' not found"
        if self.verbose:
            print(f"Trying to parse rule '{rule.element_name}' at position {position}")
        success, end_position, parsed_elements, error, matched_only_partially = rule.parse(string, position, self)
        self.memo[(rule.element_name, position)] = (
            success, end_position, parsed_elements, error, matched_only_partially)
        return success, end_position, parsed_elements, error, matched_only_partially


class Element:
    def __init__(self, element_name, element_action=None):
        self.element_name = element_name
        self.element_action = element_action

    def parse(self, string, position, grammar):
        return self.element_action(string, position, grammar) if self.element_action else None


class Rule(Element):

    def __init__(self, elements, element_name):
        super().__init__(element_name)
        self.name = element_name
        if isinstance(elements, Element):
            elements = [elements]

        self.elements = elements

    def parse(self, string, position, grammar):
        parsed_elements = []
        matched_only_partially = None
        for element in self.elements:
            success, position, element_parsed, error, matched_only_partially = grammar.parse_rule(element, string,
                                                                                                  position)
            if not success:
                return False, position, [], error, False
            parsed_elements.extend(element_parsed)
            if matched_only_partially:
                break
        return True, position, parsed_elements, None, matched_only_partially if matched_only_partially else False


class Terminal(Element):
    def __init__(self, value, element_name, partial_match_minimum_length=None, regex_terminal=False,
                 element_action=None):
        super().__init__(element_name, element_action)
        self.value = regex.compile(value) if regex_terminal else value
        self.partial_match_minimum_length = partial_match_minimum_length
        self.regex_terminal = regex_terminal

    def parse(self, string, position, grammar):
        if self.regex_terminal:
            # noinspection PyArgumentList
            match = self.value.match(string[position:], partial=True)
            if match:
                return True, position + match.end(), [match.group()], None, match.partial
            else:
                return False, position, [], f"Expected '{self.value}' at position {position}", False

        end_position = position + len(self.value)
        if string[position:end_position] == self.value:
            return True, end_position, [self.value], None, False

        if self.partial_match_minimum_length and len(string) - position >= self.partial_match_minimum_length:

            if self.value.startswith(string[position:]):
                return True, position + len(string), [string[position:]], None, True
        if self.partial_match_minimum_length and len(string) - position == 0 and position > 0:
            return True, len(string), [string[0:]], None, True
        return False, position, [], f"Expected '{self.value}' at position {position}", False


class NonTerminal(Element):
    def __init__(self, rules, element_name, element_action=None):
        super().__init__(element_name, element_action)
        if isinstance(rules, Element):
            rules = [rules]
        self.rules = rules

    def parse(self, string, position, grammar):
        end_position = position
        parsed_elements = []
        matched_only_partially = None
        for rule in self.rules:
            success, end_position, parsed_elements, error, matched_only_partially = grammar.parse_rule(rule, string,
                                                                                                       end_position)
            if not success:
                return False, position, [], f"Expected one of {self.element_name} at position {position}", matched_only_partially if matched_only_partially else False
            if matched_only_partially:
                break
        if len(self.rules) == 0:
            return False, position, [], f"Expected one of {self.element_name} at position {position}", matched_only_partially if matched_only_partially else False
        elif len(self.rules) > 0:
            return True, end_position, parsed_elements, None, matched_only_partially if matched_only_partially else False


class Choice(Element):
    def __init__(self, rules, element_name, element_action=None):
        super().__init__(element_name, element_action)
        if isinstance(rules, Element):
            rules = [rules]
        self.rules = rules

    def parse(self, string, position, grammar):
        matched_only_partially = None
        for rule in self.rules:
            success, end_position, parsed_elements, error, matched_only_partially = grammar.parse_rule(rule, string,
                                                                                                       position)
            if success:
                return True, end_position, parsed_elements, None, matched_only_partially
            if matched_only_partially:
                break
        return False, position, [], f"Expected one of {self.element_name} at position {position}", matched_only_partially if matched_only_partially else False


class Optional(Element):
    def __init__(self, rule, element_name, element_action=None):
        super().__init__(element_name, element_action)
        self.rule = rule

    def parse(self, string, position, grammar):
        success, new_position, parsed_elements, error, matched_only_partially = grammar.parse_rule(self.rule, string,
                                                                                                   position)
        if success:
            return True, new_position, parsed_elements, None, matched_only_partially
        else:
            return True, position, [], None, False


class Repeat(Element):
    def __init__(self, rule, element_name, min_repeats=0, max_repeats=None, element_action=None):
        super().__init__(element_name, element_action)
        self.rule = rule
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats

    def parse(self, string, position, grammar):
        parsed_elements = []
        repeats = 0
        while True and position < len(string):
            success, new_position, elements, error, matched_only_partially = grammar.parse_rule(self.rule, string,
                                                                                                position)
            if success:
                repeats += 1
                position = new_position
                parsed_elements.extend(elements)
                if self.max_repeats is not None and repeats >= self.max_repeats:
                    break
            else:
                break
        if repeats >= self.min_repeats:
            return True, position, parsed_elements, None, matched_only_partially if repeats > 0 else False
        else:
            return False, position, [], f"Expected at least {self.min_repeats} repeats of '{self.rule.element_name}' at position {position}", False
