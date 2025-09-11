# -*- coding: utf-8 -*-
"""
__author__ = "@eujhwang"
"""

# Do not include too generic sentences (e.g. \"Meme poster is trying to convey a message.\", \"The image is funny.\", .. etc)
FACT_DECOM_PROMPT = """
You will be given a paragraph ([Paragraph]). \
Please break down the [Paragraph] into a stringified Python list of atomic sentences.
Here are some guidelines when generating atomic sentences:
* Do not alter or paraphrase details in the original [Paragraph].
* Avoid using pronouns. Be specific when referring to objects, characters or situations.

Here is an example:
--------
[Paragraph]:
The caption, "This is the most advanced case of Surrealism I've seen.", \
is funny because it humorously treats the surreal and impossible scene (where \
a person is divided into separate body parts) as a diagnosable medical condition. \
It playfully applies the term "Surrealism," an art style known for bizarre, dreamlike imagery, \
to a clinical context. The contrast between the doctor’s serious tone and the absurd situation creates comedic irony.
[Output]:
```python
[
    "The scene is surreal.",
    "The scene is impossible.",
    "The scene shows a person divided into separate body parts.",
    "The division of body parts is presented as a diagnosis of a medical condition.",
    "Surrealism is an art style known for bizarre and dreamlike imagery.",
    "The doctor has a serious tone.",
    "The situation is absurd.",
    "The contrast between the doctor’s serious tone and the absurd situation creates comedic irony."
]
```
--------
Proceed to break down the following paragraph into a list of atomic sentences.
[Paragraph]:
{paragraph}
[Output]:
""".strip()


# Strict version of fact checking.
FACT_MATCH_PROMPT = """
Your task is to assess whether the information in [Sentence1] is present in [Sentence2]. \
[Sentence2] may consist of multiple sentences.

Here are the evaluation guidelines:
1. Mark "Yes" if [Sentence1] can be inferred from [Sentence2] -- \
whether explicitly stated, implicitly conveyed, reworded, or serving as supporting information.
2. Mark "No" if [Sentence1] is absent from [Sentence2], cannot be inferred, or contradicts it.

Proceed to evaluate.
[Sentence1]: {sentence1}
[Sentence2]: {sentence2}
[Output]:
""".strip()


# Relaxed version of FACT_MATCH_PROMPT. (We adopt this version.)
# If the model prediction generates a considerably different words that has the same meaning in the reference.
FACT_INFER_PROMPT = """
Your task is to assess whether [Sentence1] is inferable from [Sentence2]. \
[Sentence2] may consist of multiple sentences.

Here are the evaluation guidelines:
1. Mark "Yes" if [Sentence1] can be inferred from [Sentence2] -- \
whether explicitly stated, implicitly conveyed, reworded, or serving as supporting information.
2. Mark "No" if [Sentence1] is absent from [Sentence2], cannot be inferred, or contradicts it.

Proceed to evaluate.
[Sentence1]: {sentence1}
[Sentence2]: {sentence2}
[Output]:
""".strip()
