# Scratch NanoGPT Design Note
Date:

## Task definition
What exactly is the task? What does the model receive as input?
What is the expected output format?

## Dataset
Where does the data come from?
How many train / eval examples?
What did I see when I inspected 20 examples manually?

## Eval
What is the single most important metric?
How do I know if training worked?
What does failure look like?

## Architecture decisions
GPT vs BERT: why build both? What's actually different between them?
Attention: causal vs bidirectional — what breaks if you get this wrong?
MLM masking: why 80/10/10 instead of just masking everything?
NSP: why sentence pairs? What is this task actually teaching the model?

## Open questions before I start
(things I am not sure about yet)
