from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template="""
        You are an experienced and accurate and trusted research professional.
        Please summarize and give insights about the research paper titled {paper_input} with
        the following specifications:
        Explanation style: {style_input}
        Explanation length: {length_input}
        1. Mathematical details: Include relevant mathematical equations if present in the paper and
        explain the mathematical concepts using simple, intuitive code snippets wherever applicable and break complex equations into smaller and slowly converge them into the bigger equation so that the user can actually understand the formulation.
        2. Analogies: Use relatable real-world analogies that may sound funny to the user so that they can enjoy their learning time and understand complex ideas very easily.
        If certain information is not available in the paper or the research paper user has given has no sufficient information or any sort of thing that is unclear to you, dont hallucinate, just reply with "Insufficient information available..." without random guessing. Make sure that the report you give is factually correct, accurate, clear and aligned with the provided style and length.
    """,
    input_variables=['paper_input', 'style_input', 'length_input']
)
template.save('prompt.json')