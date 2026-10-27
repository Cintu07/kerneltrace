import pytest
import torch

from kerneltrace.models.adapter import wrap
from kerneltrace.tasks.ioi import IOICase
from kerneltrace.tasks.preflight import TokenizedCase

VOCAB = 64


def tiny_gpt2(attn="eager", seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=32, n_embd=32, n_layer=3, n_head=4,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    cfg._attn_implementation = attn
    return wrap(GPT2LMHeadModel(cfg).eval())


def tiny_neox(seed=0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=128, max_position_embeddings=32, hidden_dropout=0.0,
                        attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return wrap(GPTNeoXForCausalLM(cfg).eval())


def fake_cases(n=6, length=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        clean = torch.randint(4, VOCAB, (length,), generator=g).tolist()
        corrupt = list(clean)
        corrupt[length // 2] = (clean[length // 2] + 1) % VOCAB
        case = IOICase(case_id=f"c{i}", axes={}, clean="", corrupt="", answer="", distractor="",
                       split="discovery", template=i % 3)
        out.append(TokenizedCase(case, clean, corrupt, answer_id=1, distractor_id=2))
    return out


@pytest.fixture
def gpt2():
    return tiny_gpt2()


@pytest.fixture
def cases():
    return fake_cases()
