"""Group Relative Policy Optimization (GRPO), from the algorithm -- no RL framework.

Reference: DeepSeek-R1 (Jan 2025). The idea in three steps:

1. ROLLOUT: for one prompt, sample G completions from the current ("old") policy
   and score each with the verifiable reward.
2. ADVANTAGE: normalize rewards *within the group*:
       A_i = (r_i - mean(r)) / (std(r) + eps)
   No critic, no value network -- the group itself is the baseline. Completions
   better than their siblings get positive advantage, worse ones negative.
3. UPDATE: maximize the PPO-style clipped surrogate
       min( ratio * A, clip(ratio, 1-e, 1+e) * A ),   ratio = pi_new / pi_old
   averaged per completion with a 1/|o_i| length normalization (so long
   rambling answers don't dominate), minus an optional KL penalty to a
   reference policy.

What we log every step: mean reward (is it learning?), mean KL
(pi_new || pi_old, how far the policy moved), mean entropy (is it still
exploring?), mean response length.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from model import generate


@dataclass
class Rollout:
    prompt_ids: list        # token ids of the prompt (incl. <bos>)
    comp_ids: list          # token ids of the sampled completion
    old_logprobs: torch.Tensor  # per-token log p under the sampling policy


def sample_group(model, tok, prompt_text, cfg, generator):
    """Sample G completions for one prompt. Returns a list of Rollouts."""
    prompt_ids = [tok.bos_id] + tok.encode(prompt_text)
    group = []
    for _ in range(cfg.group_size):
        comp_ids, old_lp = generate(
            model, prompt_ids, cfg.max_new_tokens,
            temperature=cfg.temperature, top_k=cfg.top_k,
            generator=generator, eos_id=tok.eos_id)
        group.append(Rollout(prompt_ids, comp_ids, old_lp))
    return group


def group_advantages(rewards):
    """Group-normalized advantages. If every reward in the group is identical
    (std = 0), all advantages are 0 and the group contributes no gradient --
    nothing to learn from a tie."""
    r = torch.tensor(rewards, dtype=torch.float32)
    return ((r - r.mean()) / (r.std(unbiased=False) + 1e-4)).tolist()


def grpo_update(model, ref_model, opt, rollouts, advantages, cfg):
    """Run mu_epochs of GRPO updates over one batch of rollouts.

    Returns a dict of scalar metrics for logging.
    """
    model.train()
    eps, beta = cfg.eps_clip, cfg.kl_beta
    tot_kl = tot_ent = 0.0
    tot_len = 0
    for _ in range(cfg.mu_epochs):                       # usually 1
        opt.zero_grad()
        losses = []
        for roll, adv in zip(rollouts, advantages):
            full = torch.tensor([roll.prompt_ids + roll.comp_ids])
            P, L = len(roll.prompt_ids), len(roll.comp_ids)
            logits = model(full)                                    # (1, P+L, V)
            logp = F.log_softmax(logits[0], dim=-1)
            # Log-prob of each completion token: token at position P+t was
            # predicted from logits at position P+t-1. NOTE: we index with two
            # 1-D tensors (not a slice + list) so the result is shape (L,),
            # with entry t = logp[P-1+t, comp_ids[t]].
            rows = torch.arange(P - 1, P + L - 1)
            cols = torch.tensor(roll.comp_ids)
            new_lp = logp[rows, cols]                               # (L,)
            ratio = torch.exp(new_lp - roll.old_logprobs)           # pi_new / pi_old

            # --- clipped surrogate (the PPO idea): don't move too far per step ---
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
            pg_loss = -torch.min(s1, s2)

            # --- KL(pi_new || pi_old), k3 estimator: diagnostic even when beta=0 ---
            kl = ratio - torch.log(ratio) - 1.0

            token_loss = pg_loss
            if beta > 0 and ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(full)
                    ref_logp = F.log_softmax(ref_logits[0], dim=-1)
                    ref_lp = ref_logp[rows, cols]
                r_ref = torch.exp(ref_lp - new_lp)                  # pi_ref / pi_new
                kl_ref = r_ref - torch.log(r_ref) - 1.0              # KL(pi_new || pi_ref)
                token_loss = token_loss + beta * kl_ref

            losses.append(token_loss.mean())                        # 1/|o_i| normalization

            with torch.no_grad():                                   # diagnostics
                tot_kl += float(kl.mean())
                # True entropy: full distribution at each completion position.
                dist_logp = logp[rows]                              # (L, V)
                tot_ent += float(-(dist_logp.exp() * dist_logp).sum(-1).mean())
                tot_len += L
        loss = torch.stack(losses).mean()                           # mean over the batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

    n = len(rollouts) * cfg.mu_epochs
    return {"loss": float(loss), "kl": tot_kl / n,
            "entropy": tot_ent / n, "mean_len": tot_len / n}
