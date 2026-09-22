"""
Core pruning logic for one GSM8K sample: load the model, generate the
sample's reasoning + answer, build the input sequence, run the normal
(unmasked) forward pass, and train the mask over the thinking tokens.

pruning_eval.py imports from this file to run the whole pipeline.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_model_and_tokenizer(model_name=MODEL_NAME, device="cpu"):
    # loads the frozen base model + tokenizer, freezes all params so only
    # the mask (in train_mask) ever trains
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model


def get_sample(data, idx):
    item = data[idx]
    return item["question"], item["answer"]


def generate_reasoning_and_answer(tokenizer, model, question, device, max_new_tokens, label=""):

    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(prompt_ids)

    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, prompt_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    truncated_by_cap = False

    if "<think>" in generated_text and "</think>" in generated_text:
        _, _, after_open = generated_text.partition("<think>")
        thinking_text, _, answer_text = after_open.partition("</think>")
    else:
        # model didn't open a <think> block, or generation hit max_new_tokens
        truncated_by_cap = len(new_tokens) >= max_new_tokens
        reason = "generation hit --max_new_tokens (likely truncated mid-thought)" \
            if truncated_by_cap else "no <think>/</think> tags found in generated text"
        print(f"  [WARNING]{(' ' + label) if label else ''}: falling back to empty "
              f"thinking_text ({reason}, generated {len(new_tokens)} tokens). "
              f"This sample will be EXCLUDED from the occupancy stats.")
        thinking_text = ""
        answer_text = generated_text

    return prompt_ids[0], thinking_text.strip(), answer_text.strip(), truncated_by_cap


def make_input(tokenizer, q_tokens, thinking_text, answer_text, device):
    think_tokens = tokenizer("<think>" + thinking_text + "</think>",
                              return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    ans_tokens = tokenizer(answer_text,
                            return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    all_tokens = torch.cat([q_tokens, think_tokens, ans_tokens])
    input_ids = all_tokens.unsqueeze(0).to(device)

    think_start = len(q_tokens)
    think_end = think_start + len(think_tokens)

    return input_ids, think_start, think_end, len(think_tokens), len(ans_tokens)


def compute_base(model, input_ids, think_end, device):
    # unmasked forward pass, done once per sample
    embed_layer = model.get_input_embeddings()
    seq_len = input_ids.shape[1]
    pred_positions = torch.arange(think_end - 1, seq_len - 1, device=device)

    with torch.no_grad():
        base_embeddings = embed_layer(input_ids)
        normal_logits = model(inputs_embeds=base_embeddings).logits
        normal_probs = F.softmax(normal_logits[:, pred_positions, :], dim=-1)

    return base_embeddings, normal_probs, pred_positions


def train_mask(model, base_embeddings, think_start, think_end, normal_probs, pred_positions,
                device, lambda_sparsity, steps, lr, warmup_steps, init_score):
    # trains a mask over the thinking tokens: which ones can be zeroed out
    # while keeping the answer predictions close to the unmasked model's

    batch_size = base_embeddings.shape[0]
    num_think_tokens = think_end - think_start

    init_logit = torch.logit(torch.tensor(init_score)).item()
    mask_param = torch.full(
        (batch_size, num_think_tokens), init_logit, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([mask_param], lr=lr)

    for step in range(steps):
        warmup_progress = min(1.0, step / max(1, warmup_steps))
        lam = lambda_sparsity * warmup_progress

        optimizer.zero_grad()

        mask_soft = torch.sigmoid(mask_param)
        mask_hard = (mask_soft > 0.5).float()
        mask = mask_hard + mask_soft - mask_soft.detach()  # straight-through estimator

        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * mask.unsqueeze(-1)
        )

        masked_logits = model(inputs_embeds=embeddings).logits
        masked_log_probs = F.log_softmax(masked_logits[:, pred_positions, :], dim=-1)

        kl_per_token = F.kl_div(masked_log_probs, normal_probs, reduction="none").sum(dim=-1)
        kl_loss = kl_per_token.mean()
        sparsity_loss = mask.mean()
        total_loss = kl_loss + lam * sparsity_loss

        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_mask_soft = torch.sigmoid(mask_param)
        final_mask_hard = (final_mask_soft > 0.5).float()

    return final_mask_hard