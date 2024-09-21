
en_prompts = {
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
    "qa_gen": "Read the text below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nBe concise.",
    "sum_gen": "Summarize the following text.\n\n{context}",
    "gist": "Please summarize the core content of the following text, remove redundant information, and compress it into concise and accurate text. Retain all key facts and points. The language should be straightforward and concise, and do not use any formatting.\n\nText: {context}\n\nPlease output the core content directly.",
    "dull_reply": "I have read the article. Please provide your question."
}

zh_prompts = {
    "context": """你将获得一篇长文章。请仔细阅读这篇文章。阅读完成后，你将根据文章的内容执行特定任务。

现在，文章开始：
- **文章内容：** {context}

文章到此结束。

接下来，请按照给出的指示完成任务。""",

    "sur": """
你会得到一个与文章相关的问题。为了有效地回答这个问题，你需要回想文章中的具体细节。你的任务是生成精确的线索问题，帮助找到文章中必要的信息。

### 问题：{question}
### 指示：
1. 你对文章有一个大致的理解。你的任务是生成一个或多个具体的线索，帮助查找文章中的支持证据。
2. 线索应以精确的替代问题形式呈现，澄清原问题。
3. 只输出线索。如果有多个线索，请用换行符分隔。
4. 请用中文回答。""",

    "span": """
你会得到一个与文章相关的问题。为了有效地回答这个问题，你需要回想文章中的具体细节。你的任务是识别并提取文章中与问题相关的一个或多个具体线索文本。

### 问题：{question}
### 指示：
1. 你对文章有一个大致的理解。你的任务是生成一个或多个具体的线索，帮助查找文章中的支持证据。
2. 线索应以文本片段的形式呈现，这些片段将有助于回答问题。
3. 只输出线索。如果有多个线索，请用换行符分隔。
4. 请用中文回答。""",

    "qa": """
你会得到一个与文章相关的问题。你的任务是直接回答这个问题。

### 问题：{question}
### 指示：
基于文章的内容，直接回答问题。不要包含除答案之外的任何额外内容。""",

    "sum": """
你的任务是通过列出文章的关键点来创建一个简明的总结。每个关键点应按顺序逐行列出并编号。

### 要求：

- 关键点应简短，并着重于主要思想或事件。
- 确保每个关键点都捕捉到文章中最重要和相关的信息。
- 保持清晰连贯，确保摘要能有效传达文章的精髓。
- 请用中文回答。""",

    "qa_gen": "阅读以下文本并回答问题。\n\n{context}\n\n问题：{input}\n\n请简明扼要地回答，请使用中文回答。",

    "sum_gen": "请总结以下文本，请输出中文。\n\n{context}", 
    "gist": "请总结以下文本的核心内容，删除冗余信息，压缩为简洁、准确的文本，保留所有关键事实和要点，语言直观、简明，不要使用任何格式。\n\n文本：{context}\n\n请直接输出核心内容。",
    "dull_reply": "我已经读完文本，请提出你的问题。"
}

