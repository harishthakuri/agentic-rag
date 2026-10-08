# Diagrams for non-technical readers

| PDF | What it explains |
|---|---|
| [document-upload-flow.pdf](document-upload-flow.pdf) | How an uploaded document becomes searchable: upload, background processing, where AI is used and why |
| [question-answer-flow.pdf](question-answer-flow.pdf) | How a question gets a cited answer: the API server works with the AI models directly, while you wait |
| [rag-explained.pdf](rag-explained.pdf) | What RAG is; chunking, embeddings ("meaning fingerprints"), vectors and vector databases; how many AI models simple RAG and our Agentic RAG need; Ask vs Agent in detail |

Each PDF is printed from the self-contained HTML file next to it. To change a diagram, edit the HTML (open it in a browser to preview), then print it again with Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --print-to-pdf=docs/diagrams/question-answer-flow.pdf docs/diagrams/question-answer-flow.html
```

Icons: [Lucide](https://lucide.dev) (ISC license) and [Simple Icons](https://simpleicons.org) (CC0). FastAPI, PostgreSQL and Ollama logos are trademarks of their respective owners and identify those products only.
