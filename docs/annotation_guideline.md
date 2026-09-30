# Annotation guideline for the SRE benchmarks

This guideline was used to verify the semantic robustness evaluation (SRE) benchmarks
Charades-STA-SRE and TACoS-SRE. It corresponds to Section S1 of the supplementary material.

## Definition of semantic equivalence

A reformulation is **equivalent** to its original query if and only if it denotes the same
target moment in the same video. Concretely, all of the following must be preserved:

- the participants (who acts, and on whom or what);
- the actions and the objects they involve;
- temporal ordering between actions, where the original expresses one ("after", "then", "before");
- repetition and ordinal reference ("second", "again", "another");
- quantities ("two", "both", "a few");
- polarity ("not", "without", "never").

Only the lexical and syntactic realization may change. A reformulation that adds an action,
drops an ordinal, changes which object is acted on, or reverses polarity is meaning-altering
even if it remains fluent and topical.

## Labels

Annotators see the original query and the reformulation side by side, without the video,
and assign exactly one label:

- **Equivalent.** The reformulation satisfies the definition above.
- **Meaning-altering.** At least one property of the definition is not preserved. Record which property.
- **Ungrammatical.** The reformulation is equivalent in intent but not well formed. Supply a
  minimal correction that restores grammaticality without changing meaning; the corrected form
  replaces the generated one in the released set.

Judge meaning, not naturalness: a stilted but equivalent rewrite is *equivalent*, and a fluent
rewrite that drops "second" is *meaning-altering*.

## Worked examples

- "She cuts the pomegranate in half" -> "She combines the pieces and cuts the pomegranate":
  **meaning-altering**, because an action is added.
- "The person peels a sticker from the mango and throws it out" -> "The mango is flung out and
  a sticker is removed by the person": **meaning-altering**, because the discarded object changes.

## Sampling and adjudication

A uniformly random sample of 200 pairs is drawn from each benchmark. Two annotators label every
sampled pair independently according to this guideline. Pairs on which the two labels differ are
adjudicated by a large language model that is given the definition above, both queries, and both
annotators' labels, and asked to return one of the three labels with a one-sentence
justification. No third human annotator is involved.

## Automatic filtering before annotation

Generated candidates pass through three mechanical steps before any human involvement:

1. Candidates identical to the original after lower-casing and punctuation stripping, and
   empty candidates, are discarded.
2. The cosine similarity between the CLIP text embeddings of the original and the candidate
   must be at least 0.85. This is a coarse guard against off-topic rewrites, not evidence of
   semantic equivalence.
3. Where several of the eight candidates survive for one query, one is retained.

Variation types (synonym substitution, paraphrase, hypernym substitution, mixed) are assigned
after generation by inspecting each pair.

## Generation prompt

The prompt used to generate the reformulations (Flan-T5-XL, eight candidates per query) is
given verbatim in `docs/sre_rewrite_prompt.txt`, with `{q}` standing for the original query.
