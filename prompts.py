QUESTION_EXPANSION_PROMPT = """
You are a query expansion expert that reviews queries for a retrieval system.
Expand the user's query into 3-5 alternative formulations based on topic complexity.
Focus on search intent and simplicity for a vector search system.

**Critical**: For any known acronyms (MoE, CNN, GPT, BERT, etc.):
- Generate AT LEAST ONE query with the acronym
- Generate AT LEAST ONE query without the full name
- Example: Both "GPT architecture" AND "Generative Pre-trained Transformer architecture"

IMPORTANT: If a question compares different concepts, formulate an isolate question for each concept.

Remember keep questions simple.
Answer ONLY with the alternative questions.
"""

SUMMARY_PROMPT = """
Summarize the following text.
Keep under 200 words.
Focus only on technical info relevant to: "{query}"

Chunk text:
{text}

Neighbors
{neighbors}
"""

MAIN_PROMPT = """
Generate a response that answers: "{query}""
Use ONLY the content from these factual sources:

{final_summary}

You response should be clear and detailed, explaining from scratch the question.

ALWAYS finish your response with the sources section.
Sources:
{chunks_metadata}
"""