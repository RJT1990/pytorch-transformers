import torch
import torch.nn as nn
from pytorch_transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from torchbench.language_modelling import WikiText103

new_model = GPT2Model.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

def model_output_transform(output, target, model):
    n_embd = 1280
    vocab_size = 50257
    lm_head = nn.Linear(n_embd, vocab_size, bias=False).cuda()
    if model.config.torchscript:
        lm_head.weight = nn.Parameter(transformer.wte.weight.clone())
    else:
        lm_head.weight = model.wte.weight
    hidden_states = output[0]
    lm_logits = lm_head(hidden_states)
    return lm_logits

WikiText103.benchmark(
  model=new_model, 
  context_length=1024, 
  encoder=tokenizer, 
  model_output_transform=model_output_transform,
  paper_model_name='GPT-2 Large',
  paper_pwc_id='language-models-are-unsupervised-multitask')
