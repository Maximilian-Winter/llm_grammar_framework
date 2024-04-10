from dataclasses import dataclass
from enum import Enum
from typing import List

import requests
from sse_starlette import EventSourceResponse
from fastapi import FastAPI, Request, File, Form, UploadFile
from llama_cpp import Llama, LlamaGrammar, LogitsProcessorList

from llama_cpp_logits_processor import GrammarLogitsProcessor
from llm_grammar import LLMGrammar, Terminal, NonTerminal, Rule, Repeat, Choice, Optional
from synthetic_generation.llm_core.llama_provider import llama_generate_function

SYS_PROMPT_START_VICUNA = """"""
SYS_PROMPT_END_VICUNA = """ """
USER_PROMPT_START_VICUNA = """USER:"""
USER_PROMPT_END_VICUNA = """"""
ASSISTANT_PROMPT_START_VICUNA = """ASSISTANT:"""
ASSISTANT_PROMPT_END_VICUNA = """"""

SYS_PROMPT_START_MIXTRAL = """[INST] """
SYS_PROMPT_END_MIXTRAL = """ """
USER_PROMPT_START_MIXTRAL = """"""
USER_PROMPT_END_MIXTRAL = """ [/INST]"""
ASSISTANT_PROMPT_START_MIXTRAL = """"""
ASSISTANT_PROMPT_END_MIXTRAL = """"""

SYS_PROMPT_START_CHATML = """<|im_start|>system\n"""
SYS_PROMPT_END_CHATML = """<|im_end|>\n"""
USER_PROMPT_START_CHATML = """<|im_start|>user\n"""
USER_PROMPT_END_CHATML = """<|im_end|>\n"""
ASSISTANT_PROMPT_START_CHATML = """<|im_start|>assistant"""
ASSISTANT_PROMPT_END_CHATML = """<|im_end|>\n"""

SYS_PROMPT_START_CODE = """"""
SYS_PROMPT_END_CODE = """\n\n"""
USER_PROMPT_START_CODE = """USER:"""
USER_PROMPT_END_CODE = """\n"""
ASSISTANT_PROMPT_START_CODE = """ASSISTANT:"""
ASSISTANT_PROMPT_END_CODE = """"""


class LLMInputPromptType(Enum):
    SYS_PROMPT = 1
    USER_PROMPT = 2
    ASSISTANT_PROMPT = 3


@dataclass
class LLMInputMessage:
    message: str
    prompt_type: LLMInputPromptType


class LLMInputPromptFormatterType(Enum):
    MIXTRAL = 1
    CHATML = 2
    VICUNA = 3


@dataclass
class LLMCompletionRequest:
    prompt_formatter: LLMInputPromptFormatterType
    messages: List[LLMInputMessage]
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repeat_penalty: float
    mirostat_mode: int
    mirostat_tau: float
    mirostat_eta: float
    tfs_z: float
    stop_sequence: List[str]


class LLMInputPromptFormatter:
    def __init__(self, SYS_PROMPT_START: str, SYS_PROMPT_END: str, USER_PROMPT_START: str, USER_PROMPT_END: str,
                 ASSISTANT_PROMPT_START: str,
                 ASSISTANT_PROMPT_END: str):
        self.SYS_PROMPT_START = SYS_PROMPT_START
        self.SYS_PROMPT_END = SYS_PROMPT_END
        self.USER_PROMPT_START = USER_PROMPT_START
        self.USER_PROMPT_END = USER_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END

    def format_messages(self, messages: List[LLMInputMessage]) -> str:
        formatted_messages = ""
        for message in messages:
            if message.prompt_type == LLMInputPromptType.SYS_PROMPT:
                formatted_messages += self.SYS_PROMPT_START + message.message + self.SYS_PROMPT_END
            elif message.prompt_type == LLMInputPromptType.USER_PROMPT:
                formatted_messages += self.USER_PROMPT_START + message.message + self.USER_PROMPT_END
            elif message.prompt_type == LLMInputPromptType.ASSISTANT_PROMPT:
                formatted_messages += self.ASSISTANT_PROMPT_START + message.message + self.ASSISTANT_PROMPT_END
        return formatted_messages + self.ASSISTANT_PROMPT_START


mixtral_formatter = LLMInputPromptFormatter(SYS_PROMPT_START_MIXTRAL, SYS_PROMPT_END_MIXTRAL, USER_PROMPT_START_MIXTRAL,
                                            USER_PROMPT_END_MIXTRAL, ASSISTANT_PROMPT_START_MIXTRAL,
                                            ASSISTANT_PROMPT_END_MIXTRAL)
chatml_formatter = LLMInputPromptFormatter(SYS_PROMPT_START_CHATML, SYS_PROMPT_END_CHATML, USER_PROMPT_START_CHATML,
                                           USER_PROMPT_END_CHATML, ASSISTANT_PROMPT_START_CHATML,
                                           ASSISTANT_PROMPT_END_CHATML)
vicuna_formatter = LLMInputPromptFormatter(SYS_PROMPT_START_VICUNA, SYS_PROMPT_END_VICUNA, USER_PROMPT_START_VICUNA,
                                           USER_PROMPT_END_VICUNA, ASSISTANT_PROMPT_START_VICUNA,
                                           ASSISTANT_PROMPT_END_VICUNA)

code_formatter = LLMInputPromptFormatter(SYS_PROMPT_START_CODE, SYS_PROMPT_END_CODE, USER_PROMPT_START_CODE,
                                         USER_PROMPT_END_CODE,
                                         ASSISTANT_PROMPT_START_CODE, ASSISTANT_PROMPT_END_CODE)

start_text = '''Generate a random email address.'''

sys_prompt = '''You only can generate a random email address.'''

grammar = None
main_model = Llama(
    "../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=33,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=-1,
)
messages = [LLMInputMessage(message=sys_prompt, prompt_type=LLMInputPromptType.SYS_PROMPT),
            LLMInputMessage(message=start_text, prompt_type=LLMInputPromptType.USER_PROMPT)]


def llama_generate(model: Llama, grammar=None):
    results = []
    user_input = start_text

    llm_grammar = LLMGrammar()
    # Complex Regular Expressions

    email = Terminal(r'[a-zA-Z0-9].{5, 10}@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', 'email', regex_terminal=True)
    url = Terminal(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', 'url', regex_terminal=True)

    program = NonTerminal(
        [email], 'Test4')
    # Define rule
    llm_grammar.add_rule(Rule(program, 'PROGRAM'))
    tokenizer = model.tokenizer()

    while True:
        test_prompt = chatml_formatter.format_messages(messages)
        mirostat_mode = 2
        temp = 0.3
        tau = 4.0
        eta = 0.1
        top_k = 40
        top_p = 0.95
        llm_generated_text = ""
        print(test_prompt)
        processor = GrammarLogitsProcessor(llm_grammar, 'PROGRAM', tokenizer.decode,
                                           model.n_vocab(), False,
                                           len(tokenizer.decode(tokenizer.encode(test_prompt, add_bos=False))),
                                           model.token_eos(), 100000)
        processor_list = LogitsProcessorList([processor])
        for out in llama_generate_function(model=model, prompt=test_prompt, temperature=temp, top_k=top_k, top_p=top_p,
                                           grammar=grammar, mirostat_mode=mirostat_mode, mirostat_tau=tau,
                                           mirostat_eta=eta, logits_processor_list=processor_list,
                                           stop_sequence=["<|im_end|>"]):
            text = out['choices'][0]['text']
            llm_generated_text += text
            print(text, end="")
        print("", flush=True)

        user_input = input("USER:")
        messages.append(LLMInputMessage(message=user_input, prompt_type=LLMInputPromptType.USER_PROMPT))


llama_generate(main_model)
