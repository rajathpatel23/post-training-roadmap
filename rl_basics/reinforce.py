import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# DEBUG SWITCH
# =============================================================================
# Run tiny first:  n_episodes=2, batch_size=2
# Read ONE sample top-to-bottom before you look at averages.
# Flip DEBUG=False once the numbers feel boring.
DEBUG = True
DEBUG_EVERY = 10      # full dump on episode 0 and every N
DEBUG_N_TOKENS = 8    # 50 tokens is noise; 8 is enough to see the pattern


def _should_debug(episode: int) -> bool:
    return DEBUG and (episode == 0 or (episode + 1) % DEBUG_EVERY == 0)


def debug_header(episode: int, prompt: str, rewards: list[float]) -> None:
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    print("\n" + "=" * 60)
    print(f"DEBUG  episode {episode + 1}  prompt={prompt!r}")
    print(f"  batch rewards: {[round(r, 3) for r in rewards]}")
    print(f"  mean R = {mean_r:+.3f}   variance = {var_r:.3f}")
    print("  interpret: if variance ≈ 0, every sample looks the same to REINFORCE → tiny / noisy signal")


def debug_reward(text: str, parts: dict) -> None:
    print(f"  completion[{int(parts['n_words'])} words]: {text[:100]!r}")
    print(
        f"  reward parts:  length={parts['length']:+.2f}  "
        f"diversity={parts['diversity']:+.2f} (trigram unique={parts['unique_ratio']:.2f})  "
        f"signals={parts['signals']:+.2f} {parts['hits']}  "
        f"→ R={parts['total']:+.3f}"
    )
    print("  interpret: length band is a cheap prior; diversity fights repetition; signals are keyword hacks")


def debug_log_probs(tokenizer, completion_ids: torch.Tensor, token_log_probs: torch.Tensor, seq_log_prob: torch.Tensor) -> None:
    token_ids = completion_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    n = min(DEBUG_N_TOKENS, len(tokens))
    print(f"  shapes: completion_ids={tuple(completion_ids.shape)}  token_log_probs={tuple(token_log_probs.shape)}")
    print("  first tokens:  idx  token             logπ(a|s)    p=exp(logπ)   read-as")
    for i in range(n):
        p = float(token_log_probs[i].exp())
        if p > 0.40:
            read = "model already likes this"
        elif p > 0.05:
            read = "plausible"
        else:
            read = "surprised / rare under current π"
        print(f"  {i:3d}  {tokens[i]:<16}  {token_log_probs[i].item():9.3f}    {p:8.4f}     {read}")
    mean_lp = token_log_probs.mean().item()
    print(f"  Σ_t logπ = {seq_log_prob.item():.3f}     mean logπ = {mean_lp:.3f}")
    print("  interpret: Σ logπ grows more negative with LENGTH. Same reward + longer text → louder gradient.")
    print("             mean logπ is the length-normalized cousin — compare that across samples.")


def debug_update(reward: float, seq_log_prob: torch.Tensor, sample_loss: torch.Tensor, baseline: float | None = None, advantage: float | None = None) -> None:
    lp = seq_log_prob.item()
    if advantage is None:
        print(f"  R(τ)={reward:+.3f}  Σlogπ={lp:.3f}  loss=-({reward:+.3f}*{lp:.3f})={sample_loss.item():.3f}")
        verb = "INCREASE P(this completion)" if reward > 0 else "DECREASE P(this completion)"
        print(f"  → vanilla: {verb}")
        print("  interpret: every R>0 sample is pushed up, even the worst one in the batch. That is the variance.")
    else:
        print(
            f"  R={reward:+.3f}  b={baseline:+.3f}  A=R-b={advantage:+.3f}  "
            f"Σlogπ={lp:.3f}  loss=-({advantage:+.3f}*{lp:.3f})={sample_loss.item():.3f}"
        )
        if abs(advantage) < 1e-6:
            print("  → advantage ≈ 0: this sample was average. Almost no update. Good (low noise) and bad (no signal).")
        elif advantage > 0:
            print("  → better than batch mean → INCREASE P(this completion)")
        else:
            print("  → worse than batch mean → DECREASE P(this completion)")


def load_policy(model_name: str= 'gpt2'):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_log_probs(model: AutoTokenizer, prompt_ids:torch.Tensor, completion_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Why re-forward after generate()? generate() was under torch.no_grad(), so it
    # built no graph. REINFORCE needs ∇_θ log π, so we score the SAME tokens again
    # with grad enabled. Sample (no grad) → score (grad). That split is the design.
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    output = model(input_ids=full_ids)
    logit = output.logits
    prompt_len = prompt_ids.shape[-1]
    # Causal LM: logits[t] predicts token t+1. Completion starts at index prompt_len,
    # so we take logits[prompt_len-1 : -1] — one position LEFT of each completion token.
    # If these two lengths ever disagree, your gather is aligned to the wrong tokens.
    completion_logits = logit[:, prompt_len -1:-1,:]
    # This is log(softmax(logits)) = log π_θ(·|sₜ) for all possible tokens
    log_probs = F.log_softmax(completion_logits, dim=-1)
    # Select the log-prob of the ACTUAL token that was generated
    # This gives us log π_θ(aₜ|sₜ) for each timestep
    token_log_probs = log_probs.gather(dim=-1,index=completion_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
     # Sum = Σ_t log π_θ(aₜ|sₜ) = log π_θ(τ)
    seq_log_prob = token_log_probs.sum()
    return token_log_probs, seq_log_prob


def generate_completion(model: AutoModelForCausalLM, tokenizer:AutoTokenizer, prompt:str, max_new_tokens:int = 50, temperature:float = 1.0) -> tuple([torch.Tensor, torch.Tensor]):
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    # no_grad: sampling is an environment step, not a graph we will backprop through.
    # The graph is built later in get_log_probs on the SAME token ids.
    with torch.no_grad():
        output_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=50, pad_token_id=tokenizer.eos_token_id)

    # Split into prompt and completion
    completion_ids = output_ids[:, prompt_ids.shape[-1]:]
    completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
    return prompt_ids, completion_ids, completion_text


# =============================================================================
# COMPONENT 4: reward_fn — The reward R(τ)
# =============================================================================
# In real RLHF, this would be a trained reward model r_φ(x, y).
# Here we use simple heuristic rewards to demonstrate the algorithm.

def reward_parts(prompt: str, completion: str) -> dict:
    """Split R(τ) so you can see WHICH heuristic fired. prompt is unused — why take it?"""
    words = completion.split()
    length = 0.0
    if 10 <= len(words) <= 40:
        length = 1.0
    elif len(words) < 5:
        length = -1.0

    unique_ratio = 0.0
    diversity = 0.0
    if len(words) >= 3:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        diversity = 2.0 * unique_ratio

    helpful_signals = ["because", "therefore", "for example", "this means"]
    hits = [s for s in helpful_signals if s in completion.lower()]
    signals = 0.5 * len(hits)
    return {
        "n_words": float(len(words)),
        "length": length,
        "unique_ratio": unique_ratio,
        "diversity": diversity,
        "signals": signals,
        "hits": hits,
        "total": length + diversity + signals,
    }


def reward_fn(prompt:str, completion:str) -> float:
    return reward_parts(prompt, completion)["total"]

# =============================================================================
# COMPONENT 5: REINFORCE Training Loop
# =============================================================================
# This is where the derivation meets reality.
#
# loss = -( R(τ) · Σ_t log π_θ(aₜ|sₜ) )
#
# When PyTorch calls loss.backward(), it computes:
#   ∇_θ loss = -R(τ) · Σ_t ∇_θ log π_θ(aₜ|sₜ)
#
# Which is exactly the REINFORCE gradient (times -1 because we minimize).

def train_reinforce(n_episodes:int = 100, batch_size:int= 4, lr:float=1e-5, prompts:list[str] = None):
    """
    Full REINFORCE training loop.
 
    Math-to-code mapping:
      E_{τ~π_θ}[...]           →  average over batch_size sampled completions
      R(τ)                      →  reward_fn(prompt, completion)
      Σ_t log π_θ(aₜ|sₜ)      →  seq_log_prob (from get_log_probs)
      ∇_θ                      →  loss.backward()
      θ ← θ 
      """
    if prompts is None:
        prompts = [
            "Explain why the sky is blue:",
            "The most important thing about machine learning is",
            "To make a good decision, you should",
            "The difference between stocks and bonds is",
        ]
    print("loading model ....")
    model, tokenizer = load_policy("gpt2")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\nTraining REINFORCE for {n_episodes} episodes")
    print(f"Batch size: {batch_size} completions per prompt")
    print("=" * 60)

    all_rewards = []
    for episode in range(n_episodes):
        episode_rewards = []
        total_loss = 0.0

        prompt = prompts[episode % len(prompts)]
        # ─── ROLLOUT PHASE ───
        # Sample batch_size trajectories τ ~ π_θ
        # This is the "sampling" that estimates E[...]
        completion_data = []
        for _ in range(batch_size):
            prompt_ids, completion_ids,text = generate_completion(model, tokenizer, prompt, max_new_tokens=50)
            parts = reward_parts(prompt, text)
            reward = parts["total"]
            completion_data.append((prompt_ids, completion_ids, text, reward, parts))
            episode_rewards.append(reward)

        if _should_debug(episode):
            debug_header(episode, prompt, episode_rewards)

        # ─── POLICY GRADIENT STEP ───
        # Compute loss = -E[ R(τ) · Σ_t log π_θ(aₜ|sₜ) ]
        # The expectation E[...] is estimated by averaging over the batch

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0)
        for i, (prompt_ids, completion_ids, text, reward, parts) in enumerate(completion_data):
            token_log_probs, seq_log_probs = get_log_probs(model, prompt_ids, completion_ids)

            # ┌─────────────────────────────────────────────┐
            # │  THE REINFORCE LOSS — your derivation:      │
            # │                                             │
            # │  loss = -( R(τ) · Σ_t log π_θ(aₜ|sₜ) )       │
            # │       = -(reward · seq_log_prob)            │
            # │                                             │
            # │  Negative: we minimize loss = maximize J(θ) │
            # │  R(τ):    the scalar reward for this τ      │
            # │  seq_log_prob: Σ_t log π_θ(aₜ|sₜ)            │
            # └─────────────────────────────────────────────┘
            sample_loss = -(reward * seq_log_probs)
            batch_loss = batch_loss + sample_loss
            if _should_debug(episode):
                print(f"\n  --- vanilla sample {i + 1}/{batch_size} ---")
                debug_reward(text, parts)
                debug_log_probs(tokenizer, completion_ids, token_log_probs, seq_log_probs)
                debug_update(reward, seq_log_probs, sample_loss)

        # These four lines MUST stay inside the episode loop. Outside, you would
        # roll out 100 times and take ONE gradient step on the last batch only.
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if _should_debug(episode):
            print(f"\n  batch loss={batch_loss.item():.3f}  grad_norm(before clip)={float(grad_norm):.3f}")
            print("  interpret: loss magnitude is NOT 'how wrong you are'. It is -(R · Σlogπ).")
            print("             watch Reward / Avg(10) to see if the policy is actually improving.")
        # ─── LOGGING ───
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        all_rewards.append(avg_reward)
        if (episode + 1) % 10 == 0:
            recent_avg = sum(all_rewards[-10:]) / len(all_rewards[-10:])
            print(
                f"Episode {episode+1:>4d} | "
                f"Reward: {avg_reward:>6.2f} | "
                f"Avg(10): {recent_avg:>6.2f} | "
                f"Loss: {batch_loss.item():>8.3f}"
            )
            # Show a sample completion
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Output: {completion_data[0][2][:80]}...")
            print()
    return model, all_rewards

# =============================================================================
# COMPONENT 6: Adding a Baseline (Variance Reduction)
# =============================================================================
# Your derivation showed that subtracting b(sₜ) is unbiased because
# b doesn't depend on the ACTION. The simplest baseline is the
# average reward across the batch.

def train_reinforce_with_baseline(n_episodes:int = 100, batch_size:int= 4, lr:float=1e-5, prompts:list[str] = None):
    if prompts is None:
        prompts = [
            "Explain why the sky is blue:",
            "The most important thing about machine learning is",
            "To make a good decision, you should",
            "The difference between stocks and bonds is",
        ]
    print("loading model ....")
    model, tokenizer = load_policy("gpt2")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\nTraining REINFORCE for {n_episodes} episodes")
    print(f"Batch size: {batch_size} completions per prompt")
    print("=" * 60)
    all_rewards = []
    for episode in range(n_episodes):
        episode_rewards = []
        total_loss = 0.0
        prompt = prompts[episode % len(prompts)]
        completion_data = []
        for _ in range(batch_size):
            prompt_ids, completion_ids, text = generate_completion(model, tokenizer, prompt, max_new_tokens=50)
            parts = reward_parts(prompt, text)
            reward = parts["total"]
            completion_data.append((prompt_ids, completion_ids, text, reward, parts))
            episode_rewards.append(reward)

        # ─── BASELINE ───
        # b = mean reward across the batch
        # This is the simplest baseline — in production you'd use
        # a learned value function V(s), but the principle is identical:
        # subtract something that doesn't depend on the action
        baseline = sum(episode_rewards) / len(episode_rewards)
        if _should_debug(episode):
            debug_header(episode, prompt, episode_rewards)
            print(f"  baseline b = {baseline:+.3f}  (batch mean; same for every sample — that is why it is unbiased)")

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for i, (prompt_ids, completion_ids, text, reward, parts) in enumerate(completion_data):
            token_log_probs, seq_log_probs = get_log_probs(model, prompt_ids, completion_ids)
            # ┌──────────────────────────────────────────────────┐
            # │  REINFORCE WITH BASELINE:                        │
            # │                                                  │
            # │  advantage = R(τ) - b                            │
            # │  loss = -(advantage · Σ_t log π_θ(aₜ|sₜ))       │
            # │                                                  │
            # │  Positive advantage → make this completion       │
            # │    more likely (it was better than average)       │
            # │  Negative advantage → make it less likely        │
            # └──────────────────────────────────────────────────┘
            advantage = reward - baseline
            sample_loss = -advantage * seq_log_probs
            total_loss = total_loss + sample_loss
            if _should_debug(episode):
                print(f"\n  --- baseline sample {i + 1}/{batch_size} ---")
                debug_reward(text, parts)
                debug_log_probs(tokenizer, completion_ids, token_log_probs, seq_log_probs)
                debug_update(reward, seq_log_probs, sample_loss, baseline=baseline, advantage=advantage)

        total_loss = total_loss / batch_size
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if _should_debug(episode):
            print(f"\n  batch loss={total_loss.item():.3f}  grad_norm(before clip)={float(grad_norm):.3f}")
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        all_rewards.append(avg_reward)
        if (episode + 1) % 10 == 0:
            recent_avg = sum(all_rewards[-10:]) / len(all_rewards[-10:])
            print(
                f"Episode {episode+1:>4d} | "
                f"Reward: {avg_reward:>6.2f} | "
                f"Avg(10): {recent_avg:>6.2f} | "
                f"Loss: {total_loss.item():>8.3f}"
            )
            # Show a sample completion
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Output: {completion_data[0][2][:80]}...")
            print()
    return model, all_rewards


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Maps directly to the policy gradient derivation.")
    print("=" * 60)
    # DEBUG=True → tiny run so you can read one episode. Flip DEBUG=False for the real comparison.
    n_episodes = 2 if DEBUG else 100
    batch_size = 2 if DEBUG else 4
    print(f"DEBUG={DEBUG}  n_episodes={n_episodes}  batch_size={batch_size}")
    print("Vanilla REINFORCE:")
    model_vanilla, rewards_vanilla = train_reinforce(n_episodes=n_episodes, batch_size=batch_size, lr=1e-5)
    print("REINFORCE with baseline:")
    model_baseline, rewards_baseline = train_reinforce_with_baseline(n_episodes=n_episodes, batch_size=batch_size, lr=1e-5)

    # Compare the rewards
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    window = min(20, len(rewards_vanilla), len(rewards_baseline))
    avg_vanilla = sum(rewards_vanilla[-window:]) / window
    avg_baseline = sum(rewards_baseline[-window:]) / window
    print(f"Vanilla REINFORCE average reward: {avg_vanilla:.2f}")
    print(f"REINFORCE with baseline average reward: {avg_baseline:.2f}")
    print(f"Difference: {avg_baseline - avg_vanilla:.2f}")

    # Show the math-to-code mapping one final time
    print("\n" + "=" * 60)
    print("MATH → CODE REFERENCE")
    print("=" * 60)
    print("""
    J(θ) = E_{τ~π_θ}[R(τ)]
         → average reward over sampled completions
 
    ∇_θ J = E[ R(τ) · Σ_t ∇_θ log π_θ(aₜ|sₜ) ]
         → loss = -(reward * seq_log_prob)
           loss.backward()
 
    With baseline:
    ∇_θ J = E[ (R(τ) - b) · Σ_t ∇_θ log π_θ(aₜ|sₜ) ]
         → advantage = reward - baseline
           loss = -(advantage * seq_log_prob)
           loss.backward()
    """)
