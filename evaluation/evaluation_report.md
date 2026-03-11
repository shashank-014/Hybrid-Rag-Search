# Evaluation Report

## Strengths

- The system combines local semantic retrieval with live Tavily search, which helps cover both stable internal knowledge and recent external updates.
- Query routing keeps simple document questions away from unnecessary web calls.
- Cross-encoder reranking improves the ordering of retrieved chunks before answer generation.
- Score transparency in the UI makes it easier to inspect why a chunk was used.
- Conversation memory supports follow-up questions in the same session.

## Limitations

- FAISS scores are distance-based, so they should be interpreted as retrieval signals rather than absolute confidence.
- The current query router is heuristic-based and can still misclassify ambiguous questions.
- Document summaries are lightweight and extractive, not full abstractive summaries.
- Streaming answers still depend on OpenAI and Tavily secrets being present in Streamlit secrets.
- Wikipedia and web content quality still depends on upstream source quality.

## Future improvements

- Add offline and automated evaluation runs with precision, recall, and answer-grounding checks.
- Add citation highlighting that maps answer spans back to exact source snippets.
- Introduce query-specific dynamic context budgets instead of fixed hybrid balancing.
- Expand router logic with model-based classification once usage patterns are clearer.
- Add persistent conversation history and user-facing session export if long-running workflows become important.
