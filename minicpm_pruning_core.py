# Same pruning steps as pruning_core.py but this is for MiniCPM5
# load the model, generate the reasoning and answer for one GSM8K sample
# run the normal forward pass (no mask) and train the mask over the thinking tokens.
# Difference from Qwen: MiniCPM5's chat template already puts <think> at the end
# of the prompt so the model only writes the thinking text and </think>.
# Because of this the thinking region here is only thinking text + </think>.

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openbmb/MiniCPM5-1B"


def load_model_and_tokenizer(model_name=MODEL_NAME, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32).to(device)
    model.eval()

    # only the mask is trained
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model


def get_sample(data, idx):
    item = data[idx]
    return item["question"], item["answer"]


def generate_reasoning_and_answer(tokenizer, model, question, device, max_new_tokens, label=""):
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True,
        return_dict=False, return_tensors="pt"
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

    if "</think>" in generated_text:
        # split into thinking part and answer part
        before, _, answer_text = generated_text.partition("</think>")
        thinking_text = before.split("<think>", 1)[-1]
    else:
        # no </think> or generation stopped at max_new_tokens
        truncated_by_cap = len(new_tokens) >= max_new_tokens
        if truncated_by_cap:
            reason = "hit max_new_tokens"
        else:
            reason = "no </think> found"
        print(f"  warning {label}: {reason}, generated {len(new_tokens)} tokens, sample will be skipped")
        thinking_text = ""
        answer_text = generated_text

    return prompt_ids[0], thinking_text.strip(), answer_text.strip(), truncated_by_cap


def make_input(tokenizer, q_tokens, thinking_text, answer_text, device):
    # this don't adds <think> again if the prompt already ends with it
    prompt_tail = tokenizer.decode(q_tokens[-3:])
    if "<think>" in prompt_tail:
        think_str = thinking_text + "</think>"
    else:
        think_str = "<think>" + thinking_text + "</think>"

    think_tokens = tokenizer(think_str,
                             return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    ans_tokens = tokenizer(answer_text,
                           return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    # question + thinking + answer
    all_tokens = torch.cat([q_tokens, think_tokens, ans_tokens])
    input_ids = all_tokens.unsqueeze(0)

    think_start = len(q_tokens)
    think_end = think_start + len(think_tokens)

    return input_ids, think_start, think_end, len(think_tokens), len(ans_tokens)


def compute_base(model, input_ids, think_end, device):
    # normal forward pass without mask done once per sample
    embed_layer = model.get_input_embeddings()
    seq_len = input_ids.shape[1]

    # positions that predict the answer tokens
    pred_positions = torch.arange(think_end - 1, seq_len - 1, device=device)

    with torch.no_grad():
        base_embeddings = embed_layer(input_ids)
        normal_logits = model(inputs_embeds=base_embeddings).logits
        normal_probs = F.softmax(normal_logits[:, pred_positions, :], dim=-1)

    return base_embeddings, normal_probs, pred_positions


def train_mask(model, base_embeddings, think_start, think_end, normal_probs, pred_positions,
               device, lambda_sparsity, steps, lr, warmup_steps, init_score):
    # learn which thinking tokens can be removed while the answer
    # predictions stay close to the normal (unmasked) model

    batch_size = base_embeddings.shape[0]
    num_think_tokens = think_end - think_start

    init_logit = torch.logit(torch.tensor(init_score)).item()
    mask_param = torch.full(
        (batch_size, num_think_tokens), init_logit, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([mask_param], lr=lr)

    for step in range(steps):
        # lambda goes up slowly during warmup
        warmup_progress = min(1.0, step / max(1, warmup_steps))
        lam = lambda_sparsity * warmup_progress

        optimizer.zero_grad()

        # hard 0/1 mask in forward pass, soft mask gradient in backward pass
        mask_soft = torch.sigmoid(mask_param)
        mask_hard = (mask_soft > 0.5).float()
        mask = mask_hard + mask_soft - mask_soft.detach()

        # apply mask on thinking token embeddings
        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * mask.unsqueeze(-1)
        )

        masked_logits = model(inputs_embeds=embeddings).logits
        masked_log_probs = F.log_softmax(masked_logits[:, pred_positions, :], dim=-1)

        # KL between masked and normal predictions + sparsity
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