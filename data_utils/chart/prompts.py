prompt_thinking_reward = """
Given
<IC>: the data of an image
<Q>: a question
<A>: a reference answer
<R>: a reasoning text

Goal:
Assess whether <R> correctly and reasonably uses visible data in <IC> to justify that the correct answer to <Q> is <A>. Rate the quality as low / medium / high according to:
(a) low: Does not use data from <IC> at all, or the language is not fluent/natural, or it fails to indicate the answer to <Q> is <A>.
(b) medium: Uses data from <IC> and is written fluently, but the reasoning is overly brief or insufficiently clear.
(c) high: Uses data from <IC> and is written fluently; the reasoning progresses step by step with depth, each step is correct and reasonable; the data from <IC> appears exactly where it should; overall, the reasoning text provides very strong support that the answer to <Q> is <A>.

Example:
<IC>: [
  {"object": "bar", "attributes": ["~120k", "Q4"], "label": "Product A"},
  {"object": "bar", "attributes": ["~150k", "Q4"], "label": "Product B"},
  {"object": "bar", "attributes": ["~90k", "Q4"], "label": "Product C"},
  {"title": "Quarterly Revenue"}
]
<Q>: Which product has the highest revenue in Q4?
<A>: product b
<R>:
    [Extraction] Reads Q4 bar heights: A ~120k, B ~150k, C ~90k.
    [Calculation] Compares values: B > A and B > C.
    [Conclusion] Therefore, Product B is highest, matching the answer "product b".

<Output>: medium

According to the requirements and examples above, score the input into three categories. Please give me the result directly without any explanation or description.

<IC>: %s
<Q>: %s
<A>: %s
<R>: %s
<Output>: 
"""