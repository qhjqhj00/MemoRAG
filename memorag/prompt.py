
prompts = {
    "context": """You are provided with a long article. Read the article carefully. After reading, you will be asked to perform specific tasks based on the content of the article.

Now, the article begins:
- **Article Content:** {context}

The article ends here.

Next, follow the instructions provided to complete the tasks.""",
    "sur": """
You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to generate precise clue questions that can help locate the necessary information.

### Question: {question}
### Instructions:
1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
2. The clues are in the form of precise surrogate questions that clarify the original question.
3. Only output the clues. If there are multiple clues, separate them with a newline.""",

    "span": """
You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to identify and extract one or more specific clue texts from the article that are relevant to the question.

### Question: {question}
### Instructions:
1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
2. The clues are in the form of text spans that will assist in answering the question.
3. Only output the clues. If there are multiple clues, separate them with a newline.""",

    "qa": """
You are given a question related to the article. Your task is to answer the question directly.

### Question: {question}
### Instructions:
Provide a direct answer to the question based on the article's content. Do not include any additional text beyond the answer.""",

    "sum": """
Your task is to create a concise summary of the long article by listing its key points. Each key point should be listed on a new line and numbered sequentially.

### Requirements:

- The key points should be brief and focus on the main ideas or events.
- Ensure that each key point captures the most critical and relevant information from the article.
- Maintain clarity and coherence, making sure the summary effectively conveys the essence of the article.
""",
    "qa_gen": "Read the text below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nBe very concise.",
    "sum_gen": "Summarize the following text.\n\n{context}"
}
