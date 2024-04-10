# Define complex grammar rules
import time

from llm_grammar import LLMGrammar, Terminal, Rule, Choice, Repeat, Optional


class TestGrammar(LLMGrammar):
    def __init__(self):
        super().__init__()
        self.setup_grammar()

    def setup_grammar(self):
        # Recursive and Nested Rule Structures
        word = Terminal(r'[A-Za-z]+', 'word', regex_terminal=True)
        phrase = Rule([word, Choice([Terminal(' ', 'space'), Terminal(".", "nothing")], "choice")], 'phrase')
        clause = Rule([word, Terminal(',', 'comma'), Optional(Terminal(" ", ""), "")], 'clause')
        sentence = Rule([Repeat(Choice([clause, phrase], "sentence_piece"), element_name="sentence_pieces")],
                        'sentence')
        self.add_rule(sentence)
        self.add_rule(clause)
        self.add_rule(phrase)
        self.add_rule(word)

        # Repeating Elements
        digit = Terminal(r'\d', 'digit', regex_terminal=True)
        repeating_digits = Repeat(digit, 'repeating_digits')
        self.add_rule(repeating_digits)

        # Complex Choice Structures
        day = Terminal(r'\d{1,2}', 'day', regex_terminal=True)
        month = Terminal(r'\d{1,2}', 'month', regex_terminal=True)
        year = Terminal(r'\d{4}', 'year', regex_terminal=True)
        date_dmy = Rule([day, Terminal('/', 'slash'), month, Terminal('/', 'slash'), year], 'date_dmy')
        date_mdy = Rule([month, Terminal('/', 'slash'), day, Terminal('/', 'slash'), year], 'date_mdy')
        date = Choice([date_dmy, date_mdy], 'date')
        self.add_rule(date)

        # Complex Regular Expressions
        email = Terminal(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', 'email', regex_terminal=True)
        url = Terminal(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', 'url', regex_terminal=True)
        self.add_rule(email)
        self.add_rule(url)

        # Combining Rules for Complex Structures
        street_address = Terminal(r'\d+ [\w\s]+', 'street_address', regex_terminal=True)
        city = Terminal(r'[\w\s]+', 'city', regex_terminal=True)
        state = Terminal(r'[A-Z]{2}', 'state', regex_terminal=True)
        zip_code = Terminal(r'\d{5}', 'zip_code', regex_terminal=True)
        full_address = Rule([street_address, Terminal(', ', 'comma_space'), city, Terminal(', ', 'comma_space'), state,
                             Terminal(' ', 'space'), zip_code], 'full_address')
        self.add_rule(full_address)


# Create an instance of the TestGrammar
grammar = TestGrammar()

# Test Strings
test_strings = {
    'sentence': "The quick brown fox, jumping over the lazy dog, barked.",
    'repeating_digits': "12345",
    'date': ["27/12/2023", "12/27/2023"],
    'email': "user@example.com",
    'url': "https://www.example.com",
    'full_address': "1234 Maple Street, Springfield, IL 62704"
}


def run_tests():
    for rule_name, test_string in test_strings.items():
        start_time = time.time()
        result = None
        for i in range(1):
            if isinstance(test_string, list):
                for string in test_string:
                    result = grammar.parse(string, rule_name, verbose=True)
                    # print(f"Testing rule '{rule_name}' with '{string}': {result}")
            else:
                result = grammar.parse(test_string, rule_name, verbose=True)
                # print(f"Testing rule '{rule_name}' with '{test_string}': {result}")
        end_time = time.time()
        print(f"{result}Time taken for '{rule_name}': {(end_time - start_time) * 1000:.4f} milliseconds")


# Run the tests
run_tests()