prompt_thinking_reward = """
Given
<IC>: the data of an image
<Q>: a question
<A>: a reference answer
<R>: a reasoning text

Goal:
Assess whether <R> correctly and reasonably uses visible data in <IC> to justify that the correct answer to <Q> is <A>. Rate the quality as low / medium / high according to:
(a) low: Does not use data from <IC> at all, or the language is not fluent/natural, or it fails to indicate the answer to <Q> is <A>, or the structure is not good.
(b) medium: Uses data from <IC> and is written fluently, but the reasoning is overly brief or insufficiently clear.
(c) high: Uses data from <IC> and is written fluently; Has a very strong CoT structure (e.g. clear and logical steps); the reasoning progresses step by step with depth, each step is correct and reasonable; the data from <IC> appears exactly where it should; overall, the reasoning text provides very strong support that the answer to <Q> is <A>.

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


prompt_refine = """Given:
<IC>: the data of an image
<Q>: a question
<A>: a reference answer
<T>: a writing template

Goal:
Transform the visual information in <IC> into a textualized data description and incorporate it into a smooth, natural explanation that reasons why the correct answer to <Q> is <A>, using the format and tone defined by <T>.

Example:
<IC>: [
    {"Year": 2011, "Favorable": 0, "Unfavorable": 3.1},
    {"Year": 2012, "Favorable": 56, "Unfavorable": 38.0},
    {"Year": 2013, "Favorable": 0, "Unfavorable": 0.0},
    {"Year": 2014, "Favorable": 51, "Unfavorable": 48.0},
    {"Year": 2015, "Favorable": 0, "Unfavorable": 53.0}
]
<Q>: In which year the value was 51?
<A>: 2014
<T>:
Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]

<Output>:
Goal: Find the year in which the 'Favorable' value was 51.
Observation: The data shows the 'Favorable' values for each year are: 2011: 0, 2012: 56, 2013: 0, 2014: 51, and 2015: 0.
Reasoning: Scanning the 'Favorable' column for the number 51 leads to the corresponding year in that row.
Conclusion: The value 51 occurred in the year 2014.

Now, according to the requirements and the examples above, convert my input into the target reasoning text. Please give me the result directly without any explanation or description.

<IC>: %s
<Q>: %s
<A>: %s
<T>: %s
<Output>:
"""

prompt_template = """Analyze the preceding text (which is a Chain of Thought, or "CoT").

**Your Task:**
1.  **Evaluate Structure and Logic:** First, determine if the CoT is **well-structured** and **logical**. A "well-structured" CoT contains clear, distinct, and labeled reasoning steps (e.g., "Goal:", "Observation:", "Reasoning:", "Conclusion:").
2.  **Extract Template:** If (and only if) the CoT is **well-structured**, extract a **generic reasoning template** from it. In this template, use brackets `[ ]` to describe the general purpose of each step (as shown in Example 1).
3.  **Output None:** If the CoT is **unstructured** (e.g., it reads like a single conversational paragraph), lacks clear steps, or is logically unsound, you must output **None** (as shown in Example 2).

---

**Example 1: [Well-Structured CoT]**

**Input:**
"Goal: Find the lowest value of the red graph.\nObservation: The data for the 'Rep/Lean Rep' category across the years are: 2018: 72, 2019: 70, and 2020: 77.\nReasoning: Comparing the values, the minimum value is 70.\nConclusion: The lowest value of the red graph is 70."

**Output:**
Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]

---

**Example 2: [Poorly-Structured / Conversational CoT]**

**Input:**
I'm trying to figure out the year when the unfavorable view reaches its highest point. Looking at the data, I see that the values for each year are pretty low until 2016, where it jumps to 55. But the peak doesn't happen until 2017, when the value spikes to 64. So, it seems like the unfavorable view really hits its maximum in 2017.

**Output:**
None

Now, according to the requirements and the examples above, give me the result directly without any explanation or description.
**Input:**
%s
**Output:**
"""