# Scratch NanoGPT — From-Scratch Pretraining Report

**Status:** Runs 1-4 + SFT complete. Four DPO runs completed on the gold dataset chain: heuristic-dataset run (flat, reward hacking), gold-dataset default lr (catastrophic collapse), gold-dataset low beta (collapse persisted, different flavor — negative result), gold-dataset low learning rate (**success — first genuinely usable DPO checkpoint**, `out_dpo_gold_bpe_lowlr/tinygpt_dpo.pt`). Root cause confirmed: DPO's sequence-summed loss needs a meaningfully lower LR than SFT's per-token-mean loss at the same value — a real, reusable lesson, not just this run's fix. Next: PPO/GRPO on top of this checkpoint, and/or apply the same lr fix to the heuristic dataset.
**Why this exists:** Qwen2.5-0.5B (Project 1) was too slow to iterate on this
machine — full eval over 2,809 examples took ~1hr per direction. This is a
small transformer built from scratch (no HF/TRL) instead, small enough to
train in minutes, so every stage of the post-training pipeline (pretrain →
SFT → DPO → PPO) can be built and understood end-to-end on a model that
was never a black box to begin with.

---

## Run 1 — TinyGPT (causal, next-token)

**Config:** `configs/train/gpt_small.yaml` + `configs/model/gpt_small.yaml`
- 842,752 params — 4 layers, 4 heads, 128 embd, block_size 256
- char-level tokenizer, tiny-shakespeare (~1M chars)
- 2000 steps, batch 64, lr 3e-4, AdamW

**Result:** ~7.4 min total on this machine (M4).

| step | train loss | val loss | val ppl |
|---|---|---|---|
| 0 | 4.229 | 4.229 | 68.6 |
| 200 | 2.505 | 2.505 | 12.2 |
| 1000 | 1.909 | 1.995 | 7.4 |
| 2000 | 1.569 | 1.745 | 5.7 |

Train/val loss track together until ~step 400, then diverge (1.569 vs 1.745
by step 2000) — early overfitting, not severe, and loss hadn't fully
plateaued at 2000 steps. See `out/training_curves.png` for the full curve.

**Qualitative sample (final, unconditioned, temperature 0.8):**
```
First such abow.

Nurse:
Is in that that the peace.

RICHARD:
What quive is the might father heir,
Which down mouses on agains his crudersont, but make
Cause his provering stracts of a with learn thee,
That the that dam her win the defefore,
The disbel: make not preposess the proving to well,
Lards
```

**What it learned:**
- Dialogue format — `SPEAKER:` (caps + colon), line breaks between turns, line-start capitalization
- Common short English words: *First, such, Nurse, peace, What, might, father, heir, down, but, make, his, with, learn, thee, well*
- Shakespearean-flavored sentence shape (archaic syntax rhythm), even where the content is nonsense

**What it hasn't learned:**
- Long-range coherence / semantics — no sentence means anything
- Full word-level consistency — plausible-looking non-words throughout (*abow, quive, mouses, agains, crudersont, provering, stracts, defefore, disbel, preposess*) — the model has learned English morphology (`-ing`, `-ess`, `-ore` endings) without having locked in the actual words yet

**Read:** this is the expected qualitative stage for an ~800K-param char-level model at val loss ~1.7 on tiny-shakespeare — consistent with Karpathy's char-RNN/nanoGPT walkthroughs at similar scale/budget. Confirms the training loop, data pipeline, and architecture are all correct — not a bug signal.

**Decision:** not chasing a lower pretraining loss further right now. The goal of this exercise is understanding SFT/DPO/PPO mechanics, not literary quality — a structurally-sound-but-incoherent base model actually makes the before/after SFT contrast easier to observe, not harder.

---

## Run 2 — TinyBERT (MLM + NSP)

**Config:** `configs/train/bert_small.yaml` + `configs/model/bert_small.yaml`
- 2,430,594 params — 12 layers, 8 heads, 128 embd, block_size 128 (bumped up from the original 4/4 draft config)
- 10,000 steps, batch 32, lr 3e-4, AdamW, mlm_probability 0.15

**Result:** ~29.4 min total on this machine (M4).

| step | train MLM loss | val MLM loss | val MLM ppl | train NSP acc | val NSP acc |
|---|---|---|---|---|---|
| 0 | 4.241 | 4.248 | 70.0 | 0.523 | 0.534 |
| 2000 | 2.632 | 2.628 | 13.8 | 0.536 | 0.579 |
| 4000 | 1.815 | 1.801 | 6.05 | 0.639 | 0.620 |
| 6000 | 1.559 | 1.629 | 5.10 | 0.665 | 0.668 |
| 8000 | 1.436 | 1.497 | 4.47 | 0.644 | 0.668 |
| 10000 | 1.317 | 1.373 | 3.95 | 0.666 | 0.681 |

See `out_bert/training_curves.png` for the full curve.

**Notable pattern — NSP accuracy is gated by MLM quality, not learned independently.**
NSP sits flat at chance (~0.50-0.53) through step ~1800, even though MLM loss
is already dropping steeply (4.24 → 2.73 by then). Right around step
2000-2400 — the same window MLM loss falls fastest — NSP accuracy jumps
from ~0.55 to ~0.66 in a few hundred steps. Read: the `[CLS]` pooled
representation had nothing usable for NSP until the shared encoder's
representations crossed some quality threshold first. The two objectives
aren't learned at independent rates.

**Much less overfitting than Run 1**, despite 2.9x more params and 5x more
steps: train/val MLM loss stay within ~0.05 of each other even at step
10,000, vs. Run 1's GPT diverging by 0.18 within 2,000 steps. Likely cause:
15% *random* masking every batch is a built-in augmentation (different
corruption pattern each pass) that GPT's fixed next-token objective doesn't
get.

**Caveat — don't compare raw loss values across Run 1 and Run 2 as
"better/worse."** MLM sees bidirectional context (both sides of a masked
position); GPT only sees the left. Lower MLM loss is expected from having
more information available, not evidence BERT is a stronger model here.

**NSP accuracy is noisy, not monotonic** — oscillates ~0.60-0.71 (val) from
step 4000 on rather than climbing cleanly. `nsp_probes` / `mask_probes`
qualitative tables were logged to W&B only (not saved locally) — worth a
look to see if the two fixed NSP pairs give stable, confident predictions or
are still flip-flopping, which would clarify whether ~65-70% reflects real
learned signal or an unstable decision boundary.

---

## Run 3 — TinyGPT (BPE tokenizer)

**First attempt crashed the system — memory thrashing, not a code bug.**
Original config used `batch_size: 64` (same as Run 1's char config). The
final projection layer computes logits of shape `(batch_size, block_size,
vocab_size)` — at BPE's `vocab_size=50,257`, that's `64 × 256 × 50,257` ≈
824M elements (~3.3GB) per forward pass just for that one tensor, before
gradients/softmax/activations. Char's equivalent (`vocab_size≈66`) is ~760x
smaller. `vm_stat` showed ~80MB free and the process in uninterruptible
sleep (disk-swap thrashing) — killed it (`kill <pid>`) before it locked up
the machine further.

**Fix:** `batch_size` dropped to 8 in `configs/train/gpt_small_bpe.yaml`
(vocab size isn't a lever we can pull — it's GPT-2's real vocab — so batch
size is the one that has to give). Re-running now.

Not yet complete. Config: `configs/train/gpt_small_bpe.yaml` + `configs/model/gpt_small_bpe.yaml`.

Same data, same tokenizer-vs-char comparison framing as before — only now
also a deeper architecture (12 layers, 8 heads) than Run 1's 4/4, applied to
both Run 2 (BERT) and Run 3 (BPE) configs, so those two are comparable to
each other but not directly to Run 1's smaller architecture. GPT-2/GPT-3's
original pretrained 50,257-token byte-level BPE vocab instead of ~65 characters.

**What to watch for in the comparison:**
- Loss values aren't directly comparable across tokenizers (different vocab
  size changes the entropy floor) — perplexity and, more importantly, the
  qualitative samples are the fair comparison.
- Much bigger embedding table (50,257 × 128 ≈ 6.4M params each for `tok_emb`
  and `lm_head`, vs. ~8K for char) relative to a ~1M-character corpus — most
  BPE vocab rows will see little to no gradient signal. Expect slower
  per-step progress, and possibly less overfitting than Run 1 since there's
  more effective capacity spread thin.
- Each token carries ~4 characters on average, so `block_size=256` covers a
  much larger effective context window in BPE than in char.
- Expected upside: fewer, more meaningful "atoms" to predict (whole
  sub-words instead of single characters) could make word-level consistency
  show up faster than in Run 1, even if raw loss looks slower to move.

**Result:** 15,278,080 params, 2000 steps, `batch_size=8` (dropped from 64
after the memory-thrashing incident above), ~9.5 min total.

| step | train loss | val loss | val ppl |
|---|---|---|---|
| 0 | 10.826 | 10.824 | 50,217 |
| 200 | 5.934 | 6.062 | 429 |
| 1000 | 4.406 | 5.075 | 160 |
| 2000 | 3.838 | 4.785 | 120 |

Step-0 loss (10.83) ≈ `ln(50,257)` exactly — confirms clean random-init
entropy for a uniform 50,257-way distribution, good sanity check.

**Overfitting is much worse here than Run 1 or Run 2** — train/val gap is
~0.95 nats by step 2000, vs. ~0.18 (Run 1) and ~0.05 (Run 2). Batch 8 means
far less data diversity per step, and neither curve had plateaued yet.
Worth trying batch sizes between 8 and 64 (e.g. 16-24) in a future pass to
find a better memory/generalization tradeoff — not urgent right now.

**Qualitative sample (final, temp 0.8) — confirms the predicted upside:**
real, intact, correct Shakespeare character names throughout (*Nurse, Lucio,
Polixenes, Leontes, Buckingham, King Edward, First Murderer*), spanning
multiple plays (Measure for Measure, The Winter's Tale, Richard III) — a
much better hit rate on proper nouns/whole words than Run 1's char model,
which mangled most uncommon words into non-words. Coherence is still weak,
and the failure mode is qualitatively different: instead of letter-salad
("crudersont"), BPE blends tokens *across different real names*
("BAPTutio" ≈ Baptista + Lucio) rather than scrambling individual letters.

---

## Comparative Analysis

| | Run 1: GPT (char) | Run 2: BERT (MLM+NSP) | Run 3: GPT (BPE) | Run 4: GPT (BPE, movie-continued) |
|---|---|---|---|---|
| Params | 842,752 | 2,430,594 | 15,278,080 | 15,278,080 (same, continued) |
| Architecture | 4 layer, 4 head | 12 layer, 8 head | 12 layer, 8 head | 12 layer, 8 head |
| Vocab size | ~65+1 (`[EOS]`) | ~65+4 special | 50,257 | 50,257 |
| Steps / batch size | 2000 / 64 | 10,000 / 32 | 2000 / 8 | 4000 / 8 |
| Final train loss | 1.569 | 1.317 (MLM) | 3.838 | 3.735 (movie) |
| Final val loss | 1.745 | 1.373 (MLM) | 4.785 | 3.934 (movie) / 5.755 (Shakespeare, +0.97) |
| Train/val gap | 0.18 | 0.05 | 0.95 | 0.20 (movie) |
| Wall-clock | ~7.4 min | ~29.4 min | ~9.5 min | ~20.1 min |
| Qualitative read | Correct format, real short words, no coherence | N/A (not generative) | Correct format + intact proper nouns/names, no coherence | Fully modern style, zero Shakespeare bleed-through, weak turn-taking |

Run 4's Shakespeare-val-loss column is the forgetting check, not a normal
val loss — it's scored against a corpus the model isn't training on.

Loss values aren't comparable across rows (different objectives/vocab
sizes/context types — see caveats in each run's section above). The
train/val gap and qualitative read are the fair cross-run comparisons.

---

## Run 4 — Continued pretraining: Shakespeare → Cornell Movie-Dialogs

**Motivating idea:** does the BPE model's "poetic" Shakespeare quality
survive contact with modern dialogue, or does continued pretraining just
overwrite it? Real continued-pretraining question (same shape as adapting a
general LM to a new domain before task-tuning), and directly testable here.

**Why BPE, not char:** the char tokenizer's vocab is locked to
tiny-shakespeare's exact character set — movie dialogue would hit
`KeyError` on any unseen character. BPE (GPT-2's real vocab) covers
standard English with no such risk, and is also the tokenizer better suited
to modern text in the first place (see Run 3's caveats on Shakespeare being
a poor match for a modern-web-text BPE vocab).

**Data:** Cornell Movie-Dialogs Corpus (ConvoKit release, not the raw
scraped-script route) — 304,446 utterances, 83,032 conversations, ~19.8M
characters (~18x tiny-shakespeare). Built via `movie_data_prep.py`:
downloads the official zip, joins `speakers.json` character names onto
`utterances.jsonl` lines, reconstructs true conversation order via the
`reply-to` chain (not file order), and writes `data/movie_dialogue.txt` in
the *same* `SPEAKER:\ndialogue\n\n` block format as tiny-shakespeare.txt —
same structural convention, new domain, so the model isn't learning a new
format and a new domain at the same time.

**Infrastructure added to support this:**
- `train.py`: `data_path` (train on a different corpus — bpe only),
  `init_from` (resume from a checkpoint instead of random init),
  `secondary_eval_path` (score a second fixed corpus every eval step — the
  actual forgetting-check mechanism), `output_dir` (avoid clobbering
  `out_bpe/`).
- `bpe_data.py`: `load_data()`/`encode_corpus()` split so one tokenizer
  instance can score two different corpora without re-downloading GPT-2's
  tokenizer files.
- Config: `configs/train/gpt_bpe_movie_continued.yaml` — continues from
  `out_bpe/tinygpt_shakespeare.pt`, trains on `movie_dialogue.txt`, tracks
  `tinyshakespeare_val_loss` alongside movie val loss every eval step.
  `batch_size: 8` (same memory lesson as Run 3), `learning_rate: 1e-4`
  (lower than the from-scratch runs' 3e-4 — standard for continued
  pretraining, avoid destabilizing what's already learned), `output_dir:
  out_bpe_movie`.

**What to check once this runs:** if `tinyshakespeare_val_loss` climbs while
movie val loss drops, that's forgetting. If it stays flat or drops too,
the model is generalizing across both rather than overwriting. Either
result is useful — this is the first run in the set explicitly designed to
answer a yes/no question rather than just produce a loss curve.

**Result: real forgetting, not generalization — clean, unambiguous answer.**
4000 steps, `batch_size=8`, `lr=1e-4`, ~20.1 min.

| step | train loss (movie) | val loss (movie) | val ppl (movie) | tinyshakespeare val loss |
|---|---|---|---|---|
| 0 | 6.886 | 7.030 | 1129.9 | 4.785 |
| 200 | 4.938 | 5.075 | 159.9 | 5.339 |
| 1000 | 4.307 | 4.451 | 85.7 | 5.582 |
| 2000 | 4.010 | 4.226 | 68.5 | 5.585 |
| 3000 | 3.851 | 4.086 | 59.5 | 5.706 |
| 4000 | 3.735 | 3.934 | 51.1 | 5.755 |

Step-0 `tinyshakespeare_val_loss` (4.785) exactly matches Run 3's final val
loss — confirms `init_from` loaded the correct checkpoint.

Movie val loss dropped cleanly the whole run (7.03 → 3.93, as expected for
learning a new domain). Shakespeare val loss **climbed** from 4.785 → 5.755
(+0.97 nats, ≈2.7x higher perplexity on Shakespeare specifically) — fastest
in the first ~400 steps, then staying elevated through the rest of
training, never recovering. Both curves moving in opposite directions is
the textbook forgetting signature (see `out_bpe_movie/training_curves.png`
— visually unambiguous, the two lines diverge and never reconverge).

**Qualitative sample confirms it independently, not just the loss number.**
Final generation reads as fully modern, clipped conversational dialogue
("I'm doing here." "That's wrong." "I'mon you know!" — the last one a BPE
token-blend artifact) with zero Shakespeare bleed-through — no archaic
vocabulary, no verse rhythm, no "thee/thou." If the poetic quality had
survived, some stylistic blending would be expected; instead the style
fully flipped to the new domain. Separately: the sample shows one speaker
(`JUD`) dominating almost every turn with little real back-and-forth —
multi-speaker turn-taking isn't well learned yet, on top of the general
incoherence already seen in every run so far.

**Verdict: the motivating idea didn't pan out — a useful negative result,
not wasted effort.** The "poetic quality survives modern dialogue" thesis
is false at this scale/budget. This directly resolves the SFT base-model
decision below.

---

## Decision — which base model does SFT build on? (RESOLVED)

**Question:** char GPT (re-pretrained with `[EOS]`) or BPE GPT (after movie
continued pretraining)?

**Answer: BPE, on the `out_bpe/` checkpoint — pre-movie, not post-movie.**
Run 4 resolved the one open branch: it showed real forgetting (Shakespeare
val loss +0.97 nats, no stylistic bleed-through in the qualitative sample —
see Run 4's Result above). Per the logic laid out below (written before
Run 4 completed), that means SFT builds on the checkpoint that still has
the Shakespeare quality intact, not the one that partially overwrote it.

Reasons BPE (over char) still hold up regardless of the movie-forgetting
result:
- EOS already exists (`<|endoftext|>`) — no re-pretraining tax, unlike char
  which needs a fresh ~7-min run just to pick up the reserved `[EOS]` id.
- Better proper-noun/whole-word fidelity already observed empirically in
  Run 3 (intact character names vs. char's letter-salad) — the tokenizer
  that's actually more likely to produce usable SFT responses.

**Resolved sub-questions:**
1. **Which checkpoint:** `out_bpe/` (pre-movie). `out_bpe_movie/` is not
   used for SFT — Run 4 was a genuine, informative negative result, not a
   detour that gets quietly absorbed into the main path.
2. **SFT dataset:** stays Shakespeare-only, `sft_data.py` as already built.
   No need for a combined Shakespeare+movie set — that branch only applied
   if forgetting had come back clean.
3. **Overfitting side-note:** worth recording even though it doesn't change
   the decision — continued pretraining's train/val gap (3.735 vs 3.934,
   ≈0.20 nats at step 4000) was much smaller than Run 3's pre-movie gap
   (0.95 nats at step 2000). More data (movie corpus is ~18x tiny-shakespeare)
   likely regularizes better regardless of domain. Not going to chase this
   by training SFT on the forgetting-affected checkpoint just to get a
   better-behaved loss curve — the domain quality loss isn't worth it.

**What Run 4 was actually for, in retrospect:** answering "does continued
pretraining on a new domain overwrite an existing one at this scale/budget"
— a real, reusable finding for any future continued-pretraining attempt on
this codebase, independent of whether the specific SFT chain uses it.

---

## Design Decisions — SFT / DPO / PPO task

**Chosen behavioral target: "given a speaker, produce one complete, well-formed
utterance and stop cleanly."** Threaded through all three post-training
stages on the *same* objective, rather than three disconnected toy tasks —
so what SFT fixes vs. what DPO adds vs. what PPO adds is directly comparable
on one base model.

**Why this task:** Run 1's qualitative sample cut off mid-word ("...Lards")
— a real, observed failure, not a hypothetical one. Fixing "doesn't know
when to stop" is a concrete, motivated target.

**SFT — `sft_data.py`:**
- prompt = speaker tag (`"MENENIUS:"`), response = one complete sentence of
  that speech.
- Raw corpus lines are word-wrapped mid-sentence (checked directly — e.g.
  `"What work's, my countrymen, in hand? where go you"` continues onto the
  next raw line), so responses are built by rejoining each speech's wrapped
  lines into continuous text, then taking the first sentence split on
  `.?!` — this is what actually gives a natural stopping point, not the
  raw line boundaries.
- Zero new data sourcing (reuses tiny-shakespeare.txt only) — also sidesteps
  any tokenizer vocab mismatch risk, since it's the exact same corpus the
  char vocab was built from.
- Output: `data/sft/{train,eval}.jsonl`, `{"prompt": ..., "response": ...}` —
  same JSONL shape convention as the main repo's Project 1 SFT data.

**DPO (planned):** sample K completions per prompt from the SFT model,
score each with a cheap rule-based heuristic (ends on terminal punctuation =
good; mid-word cutoff or repeated n-gram loop = bad), take best as chosen /
worst as rejected. No LLM judge or human labeling needed — consistent with
the "verifiable reward" approach the rest of this repo already uses for
Project 3.

**PPO/GRPO (planned):** same heuristic becomes the reward function directly.

**Alternative considered, not chosen:** a Q&A/attribution task ("Who says
X? → SPEAKER") — good for demonstrating format-following specifically, but
weaker fit for DPO/PPO since correctness is binary, not preference-rankable.

**EOS token added (char tokenizer) — real stopping, not just a soft bias.**
`sft_data.py` selects well-formed responses, but `generate()` had no
stopping condition at all — always ran exactly `max_new_tokens` regardless
of content, so "well-formed" would only show up as a probability bias near
the sentence boundary, not as actually-shorter output. Fixed:
- `data.py`'s `CharTokenizer` now reserves id 0 for `[EOS]` (shifts real
  chars up by 1) — **this changes vocab_size, so `out/tinygpt_shakespeare.pt`
  is now stale and needs re-pretraining** (`python train.py --config
  configs/train/gpt_small.yaml`, ~7 min). The EOS row won't get real
  training signal until SFT — tiny-shakespeare has no document boundaries to
  place it at during pretraining, which is normal (same is true of real
  base-model EOS tokens before instruction tuning).
- `TinyGPT.generate()` now takes an `eos_id` param and breaks early once
  every sequence in the batch samples it (`max_new_tokens` remains the hard
  cap if `eos_id` is None or never sampled).
- BPE tokenizer needed nothing new — GPT-2's vocab already has
  `<|endoftext|>` (id 50256) as `eos_token_id`.
- Fixed a related bug this surfaced: the "final sanity sample" in `train.py`
  seeded generation with `torch.zeros((1,1))` (token id 0) — harmless
  before, but id 0 is now specifically `[EOS]`, so that call now seeds from
  an encoded `"\n"` instead.

---

## SFT Training

**`sft_data.py`, `sft_bpe_data.py`, `train_sft.py`, `configs/train/sft_bpe.yaml`
built.** Data: 5,710 train / 634 eval `(prompt, response)` pairs, spot-checked
manually (15 random examples printed by `sft_data.py`) — clean, real
complete sentences correctly attributed to speakers.

**First training run produced degenerate output — a real bug, not
undertraining.** Output collapsed into repeating the single highest-frequency
token (`him him him him...`, `the the the the...`, `of of of of...`),
alongside repeated punctuation (`ROMEO:::::`). More training steps would
have made this *worse*, not better — it's not a capacity/data problem.

**Root cause:** `sft_bpe_data.py`'s `build_example()` set `labels` to the
*same* token positions as `input_ids`, instead of shifted by one. A causal
LM's loss at position `t` must predict the token at `t+1` — compare to
`data.py`'s pretraining `get_batch()`, which correctly does
`y = data[i+1:i+1+block_size]`. The unshifted version trained the model
toward "predict your own current input token," a trivial objective whose
only stable optimum is repeating whatever token that pattern favors most —
exactly the observed failure.

**Fix:** `build_example()` now does the standard causal-LM shift —
`input_ids = full_ids[:-1]`, `targets = full_ids[1:]` — with the
prompt-mask applied to the *shifted* target index (`(i + 1) >= n_prompt`,
not `i >= n_prompt`), so the last prompt-token position correctly becomes a
real training signal (predicting the first response token, given the full
prompt as context) rather than being masked out.

**Result (after the fix): clean, genuine success.** 3 epochs (1000 steps),
`batch_size=16`, `lr=5e-5`, ~83s total.

| step | train loss | eval loss |
|---|---|---|
| 100 | 5.014 | 4.905 |
| 400 | 4.425 | 4.531 |
| 700 | 4.079 | 4.372 |
| 1000 | 4.198 | 4.351 |

Eval loss dropped smoothly and monotonically, converging by the end (4.360
→ 4.353 → 4.351 in the last 200 steps). Train loss is noisy (small-batch
SGD variance on ~357 steps/epoch) but that's expected, not instability.

**Qualitative — this is the actual point of the exercise, and it worked.**
Base vs. SFT on the same 4 fixed prompts, full outputs:

```
ROMEO:
  base: "ROMEO:\nAnd I will not a man.\n\nLADYea, I do you.\n\nROMEO:\nI'll not my lord.\n\n
         WARWICK:\nWhy woe no more than thou wast thou art\nThe word is a prince,\n
         I'll find your grace, my lord; and let him.\n\nSecond Senator:\nRome are it;"
  sft:  "ROMEO:No, sir, I'll be: The man's to be but I cannot stay.<|endoftext|>"

MENENIUS:
  base: "MENENIUS:\nThe tribunes are not\nIt is a husband, and my master.\n\nJULIET:\n
         You shall the duke!\n\nKING HENVOLIO:\nWhere is mine own brother!\n\nROMEO:\n
         My lord, and by my lie is the way,\nAnd what she's a very man?\nWhat, sir?\nBUCKING"
  sft:  "MENENIUS:Well, sir, I'll tell me.<|endoftext|>"

JULIET:
  base: "JULIET:\nAnd I have not thy love.\n\nGLOUCESTER:\nI fear me, good queen.\n\n
         KING RICHARD II:\nMy lord, I have, how we not.\n\nCESTER:\nAy, and the day, to
         my brother's good lord.\n\nDUKE VINCENTIO:\nAnd you are the world's"
  sft:  "JULIET:I'll give me: I'll see my lord, O thou tell you with me, for I will but
         I know he that thou to me, Which time.<|endoftext|>"

KING RICHARD III:
  base: "KING RICHARD III:\nThe king and, and with my father's death.\n\nMessenger:\n
         The prince is the day'st to help and your highness,\nAnd he shall say, but a
         thousand days\nAnd we had not.\n\nLUCIO:\n'Tis a word is,\nHe's point it you as
         a man'st\nNow, I see our husband"
  sft:  "KING RICHARD III:We have a traitor with the crown'd his friends.<|endoftext|>"
```

Every SFT response now (1) stops cleanly at `<|endoftext|>` right after one
complete utterance — zero rambling to the 300-token cap, zero mid-word
cutoffs (the exact Run 1 "...Lards" failure, fully resolved) — and (2)
stays on the single prompted speaker, never wandering into 4-6 other
characters the way the base model does on every single prompt.

**Content quality is uneven across prompts, and that's expected — not a
regression.** KING RICHARD III's response reads as genuinely
Shakespeare-flavored and complete ("crown'd" is a nice period-correct
contraction). JULIET's is weaker, grammatically wandering within the one
utterance. SFT reshapes *behavior* (format, stopping, speaker focus) — it
doesn't add understanding the base model didn't already have. The base
model's own coherence ceiling (visible since Run 1) is still the limiting
factor on content quality; fixing that isn't what this stage was for.

**Verdict: SFT stage complete and successful.** Checkpoint:
`out_sft_bpe/tinygpt_sft.pt`. Ready to move to DPO.

**Backlog — scene-conditioned dialogue (not blocking current SFT work):**
neither tiny-shakespeare.txt nor the Cornell corpus has scene descriptions
(checked directly — `grep -iE "^(ACT|SCENE) "` on tiny-shakespeare.txt
returns nothing). Adding scene context needs no architecture change (a
scene is just more prompt text, prompt-masking already handles arbitrary
prefix length) — the real work is sourcing scene-tagged text. Two options:
a synthetic proxy from Cornell's `speakers.json` movie metadata (fast, low
fidelity), or swapping to a Project Gutenberg full edition of Shakespeare
(public domain, retains real Act/Scene headers, unlike the stripped-down
tiny-shakespeare.txt currently in use) — the latter is more genuine and
worth doing if scene-conditioning becomes a priority later.

---

## Code Review / Refactor Pass (2026-07-05)

Ran `/simplify` (4 parallel review agents: reuse, simplification, efficiency,
altitude) across all of `scratch_nanogpt/` + the new `src/common/` additions.
Applied:

- **`resolve_device()`** was duplicated verbatim in `train.py`, `train_bert.py`,
  `train_sft.py` — all three now call the main repo's existing
  `src/common/generation.py::resolve_lm_device()` instead (same cascade,
  already used elsewhere in this repo — genuinely reusable, not a rewrite).
- **`sample_prompts()`** was duplicated between `train.py`/`train_sft.py` —
  extracted to a new `scratch_nanogpt/sampling.py`. `train.py`'s "final
  sanity sample" block also duplicated this inline; now calls the shared
  function too.
- **`checkpointing.py` was dead code — built but never wired in.** The
  altitude review caught the real issue: the original version conflated
  three independent concepts (which weights to train from, whether to
  continue the loss history, which checkpoint is the frozen "base" reference
  for SFT's qualitative comparison) under one `resume_from` flag, and
  wouldn't have actually fit `train.py`'s Run-4 case if wired in as-is.
  Redesigned: `init_from` now uniformly means "what training starts from /
  mutates" everywhere (matching `train.py`'s original convention);
  `continue_history` is an explicit, independent boolean; SFT gets a new,
  separate `reference_checkpoint` (defaults to `init_from`) purely for the
  qualitative base-vs-trained samples. `resume_from` as a concept is gone —
  resuming is just pointing `init_from` at the checkpoint you want to
  continue. Now actually used by both `train.py` and `train_sft.py`.
- **`exp_transform(history[-1:], ...)`** slice-of-one pattern (both
  `train.py` and `train_bert.py`) replaced with direct inline
  `math.exp(min(loss, 20.0))` — same result, no reliance on the non-obvious
  "the dict inside a 1-element slice is the same object" mutation trick.
- **`train_bert.py`'s `output_dir`** was hardcoded (the only one of the three
  training scripts without a config override) — now matches `train.py`/
  `train_sft.py`'s `cfg.get("output_dir", ...)` convention.
- **`sft_bpe_data.py` re-tokenized the full dataset on every epoch and every
  eval call** (~13x per SFT run) — added `tokenize_examples()`, called once;
  `iterate_epoch()`/`evaluate()` now reuse the cached tokenized list.
  Agent's own assessment: negligible at current data size (5,710/634
  examples), not an active bottleneck — done because it was a trivial,
  contained fix, not because it was urgent.
- Removed an unused `collate` import in `train_sft.py`.

**Explicitly not applied — real findings, deliberately deferred:**
- **No KV cache in `TinyGPT.generate()`** — the efficiency review's biggest
  finding: full sequence recomputed every decode step, ~40x more compute
  than necessary for an 80-token completion. Real, but a substantial feature
  addition (thread per-layer K/V cache through `Attention`/`Block`/
  `TinyGPT`), not a cleanup — left as a deliberate follow-up, not silently
  applied.
- **No batched generation** in `sample_prompts()` — sequential per-prompt
  `generate()` calls, even though the model already supports
  `key_padding_mask` for batching. Same reasoning: real opportunity, genuine
  feature work, compounds with the KV-cache fix once that exists.
- **`bert_data.py`'s `apply_mlm_masking`** pure-Python per-position loop —
  flagged by the efficiency review as low-priority (not unboundedly growing,
  intentionally tiny model/data) — skipped.

## Also converted configs from plain dicts to dataclasses (`config.py`)

Motivated directly by the `init_from` dual-meaning bug the altitude review
found: a plain dict lets any key mean anything under time pressure. Added
`ModelConfig`, `GPTTrainConfig`, `BERTTrainConfig`, `SFTTrainConfig` (typed,
not Pydantic — matches the main repo's own existing `src/common/config.py`
dataclass convention for the Qwen/HF pipeline; Pydantic's main value is
validating *untrusted* input, which config YAML you hand-write yourself
isn't). `train.py`/`train_bert.py`/`train_sft.py` all updated to attribute
access (`cfg.batch_size`, `cfg.model.block_size`) instead of dict indexing —
an unknown/typo'd YAML key now raises `TypeError` at load time instead of
silently doing nothing. Verified all 5 real configs still load correctly
against the new schemas. History rows and sample dicts deliberately left as
plain dicts — those vary in shape per script by design (BERT logs MLM/NSP
fields, GPT logs perplexity, SFT logs train/eval loss), forcing one rigid
schema onto genuinely different shapes would fight the data, not help it.

## Added a test suite (`scratch_nanogpt/tests/`, pytest)

Motivated by the labels-shift bug actually happening — a 5-line unit test
on `build_example()` would have caught it instantly instead of discovering
it via a degenerate training run. Added `pytest` as a dev dependency
(`pyproject.toml`'s `[project.optional-dependencies] dev`, installed via
`uv sync --extra dev`) — no existing test infrastructure in the repo before
this. 25 tests, all passing, ~0.7s runtime, no GPU/network dependency:

- `test_sft_bpe_data.py` — the labels-shift/prompt-masking logic
  specifically. Verified this suite actually has teeth: reconstructed the
  original buggy `build_example()` inline (without touching the real,
  now-fixed source) and confirmed the shift test fails against it exactly
  as expected.
- `test_bert_data.py` — segment id assignment, special-token exclusion from
  MLM masking, padding/attention-mask construction.
- `test_model.py` — `Attention`'s combined causal+padding mask actually
  blocks what it should (a later real position is unaffected by changes at
  an earlier *padded* position, even though causal masking alone would
  otherwise permit attending to it) — the kind of combinatorial masking bug
  that can silently leak information while still "training fine."
- `test_checkpointing.py` — `continue_history` behaves independently of
  anything else, per the altitude review's fix.
- `test_config.py` — every real YAML config loads against its typed
  schema; an unknown field raises `TypeError` instead of being silently
  dropped.

**Deliberately not tested:** training dynamics, generation quality, loss
convergence — those are inherently empirical/stochastic and already
validated by the qualitative before/after comparisons every training script
produces. Unit tests target the category of bug that's silent and hard to
catch from a loss curve (data/label construction, masking, config
loading), not the model's learning behavior itself.

Run: `cd scratch_nanogpt && ../.venv/bin/python -m pytest tests/ -v` (or
`uv run pytest scratch_nanogpt/tests/` from the repo root).

---

## DPO — Preference Data + Training Script

**Target: sharpen what SFT left uneven** — SFT fixed stopping/format/speaker
focus completely, but content quality still varied (KING RICHARD III's
clean output vs. JULIET's rambling one, same SFT model). DPO's job is to
push toward the better end of that variance using the SFT model's own
sample diversity as signal — no LLM judge, no human labeling, per the
design decided earlier.

**First attempt at the scoring heuristic failed almost completely — a real,
informative result, not a bug.** `dpo_data.py` samples K completions per
prompt from the SFT model and scores each with a rule-based heuristic;
best/worst become chosen/rejected. First heuristic (hit EOS + end on
punctuation + no trigram repeats) scored **3 of 287 prompts** with any
separation — 284 were skipped because every sample tied. Root cause: SFT
already solved "stop cleanly + end in punctuation" almost universally (see
SFT Training section), so a mostly-binary heuristic built around that exact
signal has nothing left to discriminate on.

**Fix:** rebuilt the heuristic around the quality axes SFT did *not* fully
solve — graduated penalties for excessive length (rambling beyond ~20
words), bigram *and* trigram repetition (not trigram-only, catches shorter
loops like "my lord, my lord"), and comma-heavy run-on structure. Bumped
`K_SAMPLES` 4 → 6 for more draws per prompt. Result: **186 pairs** (168
train / 18 eval) from 287 prompts — real, checkable signal (spot-checked
manually): chosen consistently avoids repetition ("do do do", "they are
they are", "my lord, my lord") that rejected has.

**`train_dpo.py` built** — standard DPO setup: trainable policy + frozen
reference, both initialized from `out_sft_bpe/tinygpt_sft.pt`. Reuses
`sft_bpe_data.py`'s prompt-masking convention for per-sequence log-probs
(response tokens only). Logs a **three-way** qualitative comparison
(pretrained → SFT → DPO) on the same fixed prompts every eval step — the
full trajectory across all three post-training stages, not just SFT→DPO —
plus DPO-specific diagnostics beyond the loss number: reward margin
(chosen − rejected) and preference accuracy (starts near chance, should
climb if DPO is doing anything). The loss value alone would look
identical in shape to SFT's curve; margin/accuracy are what actually show
the DPO-specific dynamic. Config: `configs/train/dpo_bpe.yaml`, `beta=0.1`
(standard default), `batch_size=8` (DPO does 2 forward passes × 2 models
per step — ~4x SFT's per-step compute at the same batch size).

**Tests added** (`test_dpo.py`, 8 tests): heuristic scoring (EOS/punctuation/
repetition/degenerate-text preference direction), DPO loss math (loss =
ln(2) exactly when policy hasn't diverged from reference — a checkable
reference point; loss decreases as the policy's margin over the
reference's grows; accuracy reflects whether reward favors chosen), and
`sequence_logprobs`' masking arithmetic against a fixed-output fake model.

**Result: mechanically correct, qualitatively did not clearly improve —
reward hacking / proxy divergence in miniature.** 3 epochs, 60 steps total
(168 train pairs, batch_size 8), `beta=0.1`, ~12s total.

| step | train loss | eval loss | train margin | eval margin | train acc | eval acc |
|---|---|---|---|---|---|---|
| 20 | 0.703 | 0.482 | 0.08 | 0.588 | 0.500 | 0.875 |
| 40 | 0.325 | 0.398 | 1.49 | 0.938 | 0.875 | 0.833 |
| 60 | 0.275 | 0.395 | 1.78 | 1.021 | 1.000 | 0.833 |

(`train_*` columns are the single last mini-batch at that step, same
snapshot convention every other script here already uses for `train_loss`
— not a full-training-set average. `train_acc=1.0` means one specific
8-example batch swept clean, not "all 168 pairs memorized.")

**Quantitative mechanics are correct**: loss dropped cleanly on both train
and eval, reward margin climbed steadily on both. **But eval accuracy
peaked at step 20 (0.875) and dipped to 0.833 for the rest of training**
while eval margin kept climbing — a train/eval divergence. With only 18
eval pairs, each flipped example swings accuracy ~5.6%, so this specific
gap is small in absolute terms, but the trend (margin still climbing while
accuracy stalls/dips) is consistent across all 3 logged points, not just
single-batch noise — a real, if mild, overfitting signature on a tiny
(168-pair) dataset against a 15M-param model.

**The qualitative comparison is the actual headline finding.** Base →
SFT → DPO on the same 4 fixed prompts:

```
ROMEO:
  sft: "I will not be gone, and see the king."           [clean, grammatical]
  dpo: "And he that I will not so, that is a bloody day."  [worse — "he that I will not so" doesn't parse]

MENENIUS:
  sft: "You hear us!"                                     [short, clean]
  dpo: "But why come a poor gentleman that is too?"       [worse — ungrammatical, longer despite length penalty]

JULIET:
  sft: "What, that knows not to do not to me?"            [already shaky]
  dpo: "I have not not not to thy knee."                  [worse — literal repeated-word artifact]

KING RICHARD III:
  sft: "O, I never swear, to give me here."
  dpo: "And I have none to my son."                       [roughly comparable — the one non-regression]
```

JULIET's DPO output containing a literal "not not not" repetition is the
most pointed detail here: the scoring heuristic explicitly penalizes
exactly this pattern during data curation (bigram/trigram repeat penalty),
and the trained policy still produced it at generation time on a new
prompt. **The model got measurably better at winning against the
heuristic (margin + accuracy both climbed, mechanically correct) without
getting better — arguably slightly worse — by the standard a human reader
would apply.** That's reward hacking / proxy-metric divergence in
miniature: optimizing a cheap rule-based proxy improved the proxy's own
score without installing the general capability (stay grammatical, don't
repeat words) the proxy was meant to stand in for.

**This is the most valuable finding in the project so far, not a failure
of the exercise.** It's an empirical, first-hand instance of exactly the
lesson the roadmap's Project 3 report was supposed to cover ("one section
on reward hacking or brittle optimization") — earned from a real training
run rather than read about secondhand.

**Candidate explanations, not yet distinguished:**
- Too few training pairs (168) / too few steps (60) for the preference
  signal to generalize past the specific training prompts.
- The heuristic rewards surface patterns (no repeat n-grams, moderate
  length) that are checkable in the curated chosen/rejected pair but don't
  correspond to a robust internal "don't repeat" capability — the policy
  may be learning to avoid repetition specifically in contexts similar to
  training prompts, not as a general rule.
- `beta=0.1` may be too permissive, letting the policy drift further from
  the reference than the (weak) preference signal actually justifies.

Not yet resolved — logged as an open question, not silently patched over.

---

## DPO Gold Dataset — Claude-authored completions vs SFT samples

Directly targets the first DPO run's root cause: the rule-based heuristic's
chosen/rejected gap was often razor-thin (both samples from the same weak
SFT model, both hitting EOS, both ending in punctuation) — nothing close to
a genuine quality gap for DPO to learn from. This dataset replaces the
"rejected" side's competitor (another SFT sample) with a real one: Claude
(Haiku, via the Anthropic API) writes the "chosen" completion instead.

**Kept prompts fixed at the same 287 already validated by `sft_data.py`,
not expanded via a wider corpus scan.** A naive regex pass over the full
`tinyshakespeare.txt` for lines ending in `:` turns up ~2,500 "speakers" —
spot-checked a sample and most are ordinary dialogue lines that happen to
end in a colon ("Cousin, farewell:", "The day frowns more and more:"), not
real speaker tags. Widening the prompt pool that way would have poisoned
the gold set with nonsense prompts. Scaled via **variants per prompt**
instead (3 gold completions per prompt, not 1) — safe axis, and avoids
pulling the policy toward memorizing one canonical phrasing per speaker
rather than the general "coherent, non-repetitive" quality the gold set is
meant to teach.

**Gold completions kept in the same short, single-clause register the SFT
data already trains on** — deliberately not elaborate multi-clause prose.
A 15M-param model can't close a gap to genuinely eloquent writing in a few
dozen DPO steps; that big a gap would just teach surface mimicry, not real
quality, and risks the same instability the too-permissive heuristic caused
in the first run.

**Built `dpo_gold_data.py`** with two generation modes and real cost
controls, given this calls a paid API:
- **Sequential** (default): one call at a time, aborts using a running
  total of ACTUAL usage (not just an upfront estimate) checked after every
  call — the strongest guarantee, since it can stop before the next call.
- **Batch** (`--batch`, Anthropic's Message Batches API, ~50% cheaper):
  submitted as one atomic job; Anthropic doesn't expose partial results
  while still processing, so there's no way to abort mid-batch on real
  spend — only a pre-submission estimate check against `--spend-cap`. A
  real limitation of the batch execution model, not an oversight; doesn't
  matter at this project's scale (low hundreds of short completions, well
  under $1 either way) but worth knowing before pointing this at a much
  larger prompt set.

Both modes cache to disk incrementally (`data/dpo_gold/gold_completions.jsonl`),
so a partial or aborted run never restarts from zero.

**Results:**
| run | mode | calls | cost | gold completions |
|---|---|---|---|---|
| 1 | sequential | 287 (1 variant/prompt) | $0.0551 | 287 |
| 2 | batch | 574 (2 more variants/prompt, reaching 3) | $0.1089 | +574 -> 861 total |

**Total: $0.164 for 861 gold completions.** Paired against 3 SFT samples
each (skipping degenerate exact ties) -> **2,583 pairs** (2,325 train / 258
eval) in `data/dpo_gold/{train,eval}.jsonl` — ~14x the first DPO run's 168
train pairs.

Spot-checked quality: gold completions stay clean, coherent, and
period-appropriate ("O, she doth teach the torches to burn bright!" for
ROMEO — Claude may be drawing on the real line here, which is fine, arguably
better). SFT "rejected" samples continue to show the same failure modes
that motivated this whole detour (repetition, ungrammatical fragments) —
now contrasted against an unambiguously better "chosen," not another
similarly-weak sample.

**Tests added** (`test_dpo_gold_data.py`, 9 tests): cost estimation, the
pending-tasks calculation shared by both generation modes (only requests
missing variants, not already-cached ones), `build_pairs` skipping
prompts without gold completions yet (spend-cap-interrupted runs) and
skipping degenerate exact ties, and that multiple gold variants per prompt
each get their own paired rejected samples.

New config: `configs/train/dpo_bpe_gold.yaml` — same `beta`/`batch_size`/
`learning_rate` as the first DPO run deliberately (the manipulated variable
across the two runs should be the data source, not hyperparameters), points
at `data/dpo_gold/` instead of `data/dpo/`, `eval_interval` scaled up for
the ~14x larger dataset.

**Result: worse regression than run 1 — reward over-optimization collapsed
generation, and the standard DPO metrics couldn't see it happening.**
3 epochs, ~870 steps, `beta=0.1` (unchanged from run 1 — data source was
the only manipulated variable), ~133s total.

| metric | mid-run (step 350) | final |
|---|---|---|
| train loss | 0.049 | 0.00011 |
| eval loss | 0.020 | 0.0145 |
| train margin | 6.60 | 10.37 |
| eval margin | — | 8.80 |
| eval accuracy | 0.992 | 0.996 |

Eval accuracy saturated near-perfect by roughly a third of the way through
training (flagged mid-run as a concern) and the margin never stopped
climbing afterward — ending nearly 2x higher than the step-350 checkpoint.
`train_loss=0.00011` is essentially zero: the policy fit the training
preference pairs almost exactly.

**Qualitative comparison, same 4 fixed prompts as every other run:**

```
ROMEO:
  sft: "I will not be gone, and see the king."
  dpo: "I'll imitate yawn castlesel you watch my rage here beheld my son is"

MENENIUS:
  sft: "You hear us!"
  dpo: "I dare MEer for such a letter for 'tis power."

JULIET:
  sft: "What, that knows not to do not to me?"
  dpo: "I hope tackruitsblege for such aninks my woes hadruchio in arms That Henry night off as it in day and"

KING RICHARD III:
  sft: "O, I never swear, to give me here."
  dpo: "Heixals heaven with his power they broile my foe"
```

Every single DPO output is worse than its SFT counterpart — not "not
clearly better" like run 1, an unambiguous regression. `"castlesel"`,
`"tackruitsblege"`, `"hadruchio"`, `"MEer"`, `"Heixals"` are broken,
non-word BPE token sequences — the same category of failure as Run 1's
char-level `"crudersont"`, now showing up at the BPE level instead.

**Root cause: DPO's loss/margin/accuracy are all computed under teacher
forcing** (the model is fed the correct prior tokens at every step and
just ranks log-probs) — that can look perfect (0.996 accuracy) while
actual autoregressive generation, where the model conditions on its own
previous output instead of ground truth, has collapsed. Chasing an
unbounded margin (nothing in the loss stops rewarding further separation
once accuracy is already saturated) pushed the policy far enough from the
SFT reference that its free-running distribution broke down into token
soup — classic **exposure bias**, and a genuinely important methodological
finding: **the standard DPO training metrics cannot detect this failure
mode at all.** Only the qualitative sampling comparison catches it, which
is exactly why that comparison is built into every run in this pipeline
rather than trusting the loss curve alone.

**This checkpoint (`out_dpo_gold_bpe/tinygpt_dpo.pt`) is not usable as-is —
deleted 2026-07-05 so it can't get loaded as an `init_from`/reference by
mistake in a later run.** `training_curves.png`, `train_history.json`, and
the wandb run (linked above) are kept as the diagnostic record; re-running
`train_dpo.py --config configs/train/dpo_bpe_gold.yaml` regenerates the
checkpoint if needed again (e.g. after a beta/early-stopping fix).
Leading candidate fixes, not yet tried:
- Early stopping once eval accuracy saturates, instead of a fixed 3
  epochs — the margin-explosion phase (everything past ~step 300-350) may
  be pure downside with no offsetting benefit.
- Lower `beta` — 0.1 was apparently too permissive once the discrimination
  task became this easy; a smaller beta anchors the policy closer to the
  reference for the same preference margin.
- Length-normalizing the objective (SimPO-style) so the loss can't be
  satisfied just by driving raw summed log-prob margin arbitrarily high.

**Comparing the two DPO runs so far:** run 1 (heuristic dataset) was
mechanically correct but qualitatively flat — a subtler reward-hacking
result. Run 2 (gold dataset, same hyperparameters) is mechanically "more
correct" by every logged metric and qualitatively *catastrophic*. The
easier and more separable the preference task, the more dangerous
unconstrained margin-chasing becomes — a real, load-bearing lesson for
picking DPO hyperparameters, not just data.

## Diagnosing the gold-run collapse — infra improvements + a negative result on beta

Added two things to `train_dpo.py` before investigating further, both
motivated directly by run 2's collapse:
- **Multi-sample qualitative eval** (`sampling.py::sample_prompts_multi`,
  k=3 draws per prompt instead of 1) — a single stochastic draw at
  `temperature=0.8` can't tell "the policy degraded" apart from "unlucky
  low-probability sample."
- **Intermediate checkpoints** saved at every eval step (`checkpoints/step_N.pt`),
  not just the final one — so a good stopping point, if one exists, is
  actually recoverable.

**Sanity check first: is this a pre-existing BPE weakness, or did DPO cause
it?** 20 stochastic samples (5 trials x 4 prompts) from the untouched SFT
checkpoint at the same temperature/top_k produced **zero** broken outputs.
This rules out "BPE at this scale is just fragile" — the reference point is
demonstrably clean. Whatever broke, DPO training did it.

**Beta sweep result: negative.** Re-ran the gold dataset with `beta=0.02`
(5x lower) — `configs/train/dpo_bpe_gold_lowbeta.yaml`. Margin and accuracy
both came down as expected (final eval_margin ~5.2 vs 8.80; eval_acc ~0.977
vs 0.996), confirming beta does control those numbers. **But the qualitative
collapse did not go away — it changed shape.** Instead of malformed
archaic-word blends ("castlesel", "tackruitsblege"), the low-beta run
produces fluent-looking sentences contaminated with completely unrelated,
modern/off-topic vocabulary: `"cryptocam boutique"`, `"controversial"`,
`"Fahrenheit"`, `"Zimbabwe"`, `"legislatures"`, `"Cyborg"`, random numbers
(`"323"`, `"137"`, `"472"`).

**Critical detail from tracing all 3 draws across every logged step (not
just the final checkpoint): this contamination is present from step 50
onward** — roughly 6% into training, well before eval accuracy even
saturates (~step 250-300 in both runs). This rules out the "trained past
the point of real signal" story as the primary mechanism — the problem
starts almost immediately, not late. **Beta was not the right lever.**
(Checkpoint deleted — same reasoning as run 2, confirmed broken from
the multi-sample trace, diagnostic curves/history kept.)

**Working hypothesis, not yet confirmed:** SFT's cross-entropy loss
averages over tokens (`reduction='mean'`) — many tokens per batch share
and dilute the gradient signal. DPO's loss sums log-probs over the whole
response sequence (see the loss-aggregation discussion + `wiki/concepts/dpo.md`)
— a structurally larger-magnitude gradient per step, at the *same*
`learning_rate=5e-5` tuned for SFT's averaged loss. Combined with this
model's extreme vocab-to-hidden-dim ratio (50,257 vocab / 128 `n_embd` —
very compressed), an oversized update pushing up Shakespeare-vocabulary
logits could easily spill onto nearby, never-suppressed directions in that
small space — which is exactly what unrelated, rarely-used tokens would
look like from initialization, since pretraining never had to explicitly
push them down (they never appeared in Shakespeare text). Testing this
directly: `configs/train/dpo_bpe_gold_lowlr.yaml`, same gold dataset,
`beta` back to 0.1, `learning_rate=5e-6` (10x lower).

**Result: confirmed. Learning rate was the actual lever, not beta —
this run produced the first genuinely usable DPO checkpoint.** Final
metrics stayed sane throughout: `eval_acc=0.955` (never saturated to
0.99+ the way both prior runs did), `train_loss=0.140` (nowhere near the
~0 collapse of the beta=0.1/lr=5e-5 run), `eval_margin=3.33` (climbing
gently, not exploding). Qualitative comparison on the same 4 fixed
prompts — genuinely coherent Shakespeare-register dialogue, correctly
using real character names and relationships:

```
ROMEO:  "I dare assure thee for time that Montague and victory!"
JULIET: "The moon of Edward's queen's blood of their blood, and Margaret
         was patient: For triumph is England hast made their heads and
         slaughter'd me down the Tower."
KING RICHARD III: "But Richard Stanley shall make that Warwick from my
         heart the world thy words in mine eyes dost Plantagenet is brief."
```

No broken tokens, no off-topic vocabulary contamination. A handful of
minor rough contractions remain ("amll'd", "Iiolanus", "Howll'd") — this
isn't perfect, but it's a different category of problem entirely from the
prior two runs' collapses, and arguably richer/more dramatically vivid
than the terser SFT baseline while staying in-register.

**Root cause confirmed:** SFT's `reduction='mean'` cross-entropy loss
spreads gradient signal across every token in the batch; DPO's
sequence-summed log-prob objective (see loss-aggregation discussion above
and `wiki/concepts/dpo.md`) produces a structurally larger-magnitude
gradient per step at the same learning rate. Both prior collapses (beta=0.1
and beta=0.02, both at lr=5e-5) showed contamination from step 50 onward —
this was never really about training too long or beta being too permissive;
it was oversized gradient steps from the very first update, and lowering
beta doesn't reduce the gradient step size nearly as directly as lowering
the learning rate does. **General, reusable lesson: DPO's loss shape means
it likely needs a meaningfully lower learning rate than SFT by default, not
just as a tuned hyperparameter afterthought** — worth carrying into any
future DPO work (including a re-run of the run 1 heuristic dataset, and
any future PPO/GRPO stage, though PPO/GRPO's objective differs enough that
this specific numeric lr isn't necessarily transferable, just the
general "check gradient-magnitude-vs-loss-shape" principle).

Checkpoint kept: `out_dpo_gold_bpe_lowlr/tinygpt_dpo.pt` — first DPO
checkpoint from this project actually worth using downstream.

### At a glance: the whole DPO journey, one prompt at a time

Same 4 fixed prompts across every stage — pretrained and SFT are constant;
DPO shown for all three gold-dataset attempts, so the failure -> failure ->
fix progression is visible directly, not just implied by metrics.

**ROMEO:**
- pretrained: *"And I will not a man... LADYea, I do you... I'll not my lord..."* (rambling, breaks character)
- SFT: *"I will not be gone, and see the king."*
- DPO default (beta=0.1, lr=5e-5): *"I'll imitate yawn castlesel you watch my rage here beheld my son is"* — broken tokens
- DPO low-beta (beta=0.02, lr=5e-5): *"Announce kingrophe me"* — off-topic vocabulary contamination
- **DPO low-lr (beta=0.1, lr=5e-6): "I dare assure thee for time that Montague and victory!"** — coherent, correct character name

**MENENIUS:**
- pretrained: *"The tribunes are not / It is a husband, and my master..."* (incoherent)
- SFT: *"You hear us!"*
- **DPO low-lr: "I warrant you take good sir, take my father withal."** — clean, grammatical

**JULIET:**
- pretrained: *"And I have not thy love... I fear me, good queen..."* (incoherent, wrong speaker voice)
- SFT: *"What, that knows not to do not to me?"* (a bit broken)
- **DPO low-lr: "The moon of Edward's queen's blood of their blood, and Margaret was patient: For triumph is England hast made their heads and slaughter'd me down the Tower."** — genuinely dramatic, references real characters (Edward, Margaret) and events (the Tower) coherently

**KING RICHARD III:**
- pretrained: *"The king and, and with my father's death..."* (fragmented)
- SFT: *"O, I never swear, to give me here."*
- **DPO low-lr: "But Richard Stanley shall make that Warwick from my heart the world thy words in mine eyes dost Plantagenet is brief."** — a bit run-on grammatically, but in-register and correctly name-drops real characters (Stanley, Warwick, Plantagenet)

**Verdict:** the low-lr DPO run is the first checkpoint in this whole
project that's an unambiguous, genuine improvement over SFT rather than
flat or actively broken — richer and more dramatically vivid, correctly
using real Shakespeare character names/relationships that never appeared
in the terser SFT outputs. Not flawless (KING RICHARD III's line runs on;
multi-sample draws showed occasional minor garbled contractions like
"amll'd", "Iiolanus") but a real, working result.

### Three preference-data techniques, worth keeping straight

There are now three genuinely different ways to build the (prompt, chosen,
rejected) triples DPO needs, sitting at different points on the same
tradeoff:

1. **Heuristic pointwise scoring** (`dpo_data.py`, first DPO run) — both
   candidates are SFT-model samples; a hand-written rule scores each
   independently, best/worst become chosen/rejected. Cheap, deterministic,
   but blind to anything the rule doesn't encode — this is what produced
   the reward-hacking result (see DPO Result above).
2. **LLM as author** (`dpo_gold_data.py`, this section) — Claude writes the
   "chosen" response directly; "rejected" stays an SFT sample. Guarantees a
   real quality gap, but "chosen" comes from a different capability tier
   than the model being trained, not the SFT model's own ceiling.
3. **LLM as pairwise judge** (not yet built) — both candidates stay SFT
   samples (like #1), but an LLM judge picks the better one instead of a
   rule scoring them independently. Keeps both sides in-distribution (same
   weak model, same capability ceiling) while replacing the crude
   heuristic with actual judgment — the most faithful RLAIF-style setup of
   the three, since the model would be learning to prefer its own better
   outputs over its own worse ones, just with a smarter referee than a
   hand-written rule. Worth building as a follow-up once the gold-dataset
   run's results are in, to see whether the reward-hacking result was about
   the *judgment mechanism* (rule vs. LLM) or the *quality gap* (weak vs.
   strong candidate pool) — the gold dataset changes both at once, so it
   alone won't distinguish which lever mattered.

---

## Next

- [x] Run TinyBERT (MLM+NSP) — see Run 2
- [x] Run TinyGPT (BPE) — see Run 3
- [x] Fill in the comparative analysis table
- [x] Run Run 4 — BPE continued pretraining on movie dialogue — real forgetting, see Result above
- [x] Pick the SFT base checkpoint — resolved: `out_bpe/` (pre-movie), see Decision above
- [x] SFT dataset stays Shakespeare-only, no combined set needed (forgetting check came back non-clean)
- [x] Run `sft_data.py`, manually inspect the 15 sample pairs it prints — clean
- [x] Build SFT training script — hit + fixed the labels-shift bug, see SFT Training above
- [x] Re-run SFT with the fix — clean success, see Result above
- [x] Code review pass (`/simplify`) — reuse/simplification/efficiency/altitude, fixes applied, see Code Review section
- [x] Convert configs to dataclasses — catches the exact class of bug (typo'd/dual-meaning config keys) the review surfaced
- [x] Add pytest test suite (25 tests) — verified it would have caught the labels-shift bug on the first run
- [x] Build DPO preference data (`dpo_data.py`) — fixed a too-coarse heuristic (3→186 pairs), see DPO section
- [x] Build `train_dpo.py` + tests (34 tests total now) — 3-way qualitative comparison, reward margin/accuracy diagnostics
- [x] Run `train_dpo.py --config configs/train/dpo_bpe.yaml` — mechanically correct, but reward-hacking/proxy-divergence result, see DPO Result above
- [x] Build gold DPO dataset (`dpo_gold_data.py`) — Claude-authored completions vs SFT samples, 2,583 pairs, $0.164 total, see DPO Gold Dataset section
- [x] Run `train_dpo.py --config configs/train/dpo_bpe_gold.yaml` — worse than run 1: confirmed reward over-optimization, qualitative collapse (broken BPE token soup), checkpoint deleted, see DPO Gold Dataset Result above
- [x] Beta sweep on the gold dataset (beta=0.02) — negative result: margin/accuracy came down but qualitative collapse persisted in a different form (off-topic vocabulary contamination), and traced to being present from step 50 onward — too early to be over-optimization, beta wasn't the lever
- [x] Added multi-sample qualitative eval (k=3 draws/prompt) + intermediate checkpoint saving to `train_dpo.py` — needed to properly diagnose the above, not just cosmetic
- [x] Learning-rate experiment on the gold dataset (5e-6, 10x lower, beta back to 0.1) — **confirmed root cause and produced the first usable DPO checkpoint**, see Result above
- [ ] PPO/GRPO recipe on top of `out_dpo_gold_bpe_lowlr/tinygpt_dpo.pt` (first usable DPO checkpoint)
- [ ] Apply the same lr fix (5e-6) to the run-1 heuristic dataset — worth checking whether it was silently in the same over-aggressive-gradient regime, just masked by a smaller quality gap
- [ ] Backlog: KV-cache in `TinyGPT.generate()` + batched `sample_prompts()` — real ~40x generation speedup available, deferred as feature work not cleanup (see Code Review section)
- [ ] Backlog: scene-conditioned dialogue (Gutenberg Shakespeare edition with real Act/Scene headers, or a Cornell movie-metadata synthetic proxy) — not blocking, see SFT Training section
- [ ] Low priority / optional: char pretraining re-run (`configs/train/gpt_small.yaml`) if the char path is ever revisited later — vocab_size changed with the `[EOS]` token, `out/tinygpt_shakespeare.pt` is stale, but char isn't the chosen SFT path so this isn't blocking anything
