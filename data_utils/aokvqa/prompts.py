prompt_thinking_reward = """
Given
<C>: A collection of factual snippets or context.
<Q>: a question
<A>: a reference answer
<R>: a reasoning text

Goal:
Assess whether <R> correctly and logically uses the provided information in <C> to justify that the correct answer to <Q> is <A>. Rate the quality as low / medium / high according to:
(a) low: Does not use information from <C> at all, or the information used is irrelevant, or the language is not fluent/natural, or it fails to indicate the answer to <Q> is <A>, or the structure is not good.
(b) medium: Uses relevant information from <C> and is written fluently, but the reasoning is overly brief or insufficiently clear to fully connect the facts to the answer.
(c) high: Uses relevant information from <C> and is written fluently; Has a very strong CoT structure (e.g. clear and logical steps); the reasoning progresses step by step with depth, each step is correct and logical; the information from <C> is cited or referenced correctly and appropriately; overall, the reasoning text provides very strong support that the answer to <Q> is <A>.

Example:
<C>: [
  {"snippet": "The sky appears blue due to a phenomenon called Rayleigh scattering."},
  {"snippet": "Rayleigh scattering causes shorter wavelengths of light, like blue and violet, to be scattered more than longer wavelengths, like red and yellow."},
  {"snippet": "The human eye is more sensitive to blue light than to violet light."}
]
<Q>: Why is the sky blue?
<A>: The sky is blue because air molecules scatter blue light more than other colors.
<R>:
    [Fact Extraction] The context states the sky's blue color is due to Rayleigh scattering, which scatters shorter wavelengths (like blue) more effectively.
    [Synthesis] This means that as sunlight enters the atmosphere, the blue light is scattered all around, making the sky appear blue to us.
    [Conclusion] Therefore, the sky is blue because the air scatters blue light more than other colors, which aligns with the answer.

<Output>: medium

According to the requirements and examples above, score the input into three categories. Please give me the result directly without any explanation or description.

<C>: %s
<Q>: %s
<A>: %s
<R>: %s
<Output>: 
"""

prompt_refine = """Given:
<C>: A collection of factual snippets or context.
<Q>: a question
<A>: a reference answer
<T>: a writing template

Goal:
Synthesize the factual information in <C> into a smooth, natural explanation that reasons why the correct answer to <Q> is <A>, using the format and tone defined by <T>.

Example:
<C>: [
    {"fact": "The human heart is an organ that pumps blood throughout the body via the circulatory system."},
    {"fact": "This process supplies oxygen and nutrients to the tissues and removes carbon dioxide and other wastes."}
]
<Q>: What is the main function of the heart?
<A>: To pump blood throughout the body.
<T>:
Initial Premise: [Start with the most direct fact from the context.]
Elaboration: [Provide more detail or explain the 'how' or 'why' based on other facts.]
Conclusion: [Directly answer the question by summarizing the reasoning.]

<Output>:
Initial Premise: The primary function of the heart is to pump blood throughout the body using the circulatory system.
Elaboration: By pumping blood, it ensures that all body tissues receive the oxygen and nutrients they need to function, while simultaneously removing waste products like carbon dioxide.
Conclusion: Therefore, the main function of the heart is to act as the central pump for the body's circulatory system.

Now, according to the requirements and the examples above, convert my input into the target reasoning text. Please give me the result directly without any explanation or description.

<C>: %s
<Q>: %s
<A>: %s
<T>: %s
<Output>:
"""

prompt_template = """Analyze the preceding text (which is a Chain of Thought, or "CoT").

**Your Task:**
1.  **Evaluate Structure and Logic:** First, determine if the CoT is **well-structured** and **logical**. A "well-structured" CoT contains clear, distinct, and labeled reasoning steps (e.g., "Objective:", "Fact 1:", "Reasoning:", "Conclusion:").
2.  **Extract Template:** If (and only if) the CoT is **well-structured**, extract a **generic reasoning template** from it. In this template, use brackets `[ ]` to describe the general purpose of each step (as shown in Example 1).
3.  **Output None:** If the CoT is **unstructured** (e.g., it reads like a single conversational paragraph), lacks clear steps, or is logically unsound, you must output **None** (as shown in Example 2).

---

**Example 1: [Well-Structured CoT]**

**Input:**
"Objective: To explain why leaves change color in the fall.
Fact 1: Leaves contain chlorophyll, which makes them green and is used for photosynthesis.
Fact 2: As days get shorter, trees stop producing chlorophyll.
Reasoning: When the green chlorophyll fades, other pigments already in the leaves, like yellows and oranges, become visible.
Conclusion: Therefore, leaves change color because the dominant green pigment disappears, revealing the colors that were there all along."

**Output:**
Objective: [State the question's goal]
Fact 1: [Present the first relevant piece of information]
Fact 2: [Present the second relevant piece of information]
Reasoning: [Explain how the facts connect to each other logically]
Conclusion: [Provide the final answer that synthesizes the reasoning]

---

**Example 2: [Poorly-Structured / Conversational CoT]**

**Input:**
"Well, you see, the reason we have seasons is because the Earth is tilted on its axis. It's not about being closer or farther from the sun. As the Earth orbits the sun, sometimes the Northern Hemisphere is tilted toward the sun, so it gets more direct sunlight and it's summer. Then later, it's tilted away, and it's winter. That's pretty much it."

**Output:**
None

Now, according to the requirements and the examples above, give me the result directly without any explanation or description.
**Input:**
%s
**Output:**
"""