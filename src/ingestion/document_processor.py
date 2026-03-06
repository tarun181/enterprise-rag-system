import os
import re
from typing import List, Dict
import yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter


class DocumentIngestor:
    def __init__(self, config_path: str = "configs/retriever.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["chunking"]

        # Recursive splitter is excellent for code/markdown preservation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", "```", "(?<=\. )", " ", ""],
            is_separator_regex=True  # <-- ADDED THIS: Required for the regex lookbehind to work!
        )

    def process_html_file(self, file_path):
        # Define the structural elements we want to group by
        headers_to_split_on = [
            ("h1", "Title"),
            ("h2", "Section"),
            ("h3", "Subsection"),
        ]

        # 1. Split structurally first
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        structural_chunks = html_splitter.split_text_from_file(file_path)

        # 2. Then apply a smaller recursive splitter just to ensure
        # no single section exceeds the embedding model's context window
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks since metadata handles context now
            chunk_overlap=50
        )

        final_chunks = text_splitter.split_documents(structural_chunks)
        return final_chunks

    def process_directory(self, raw_docs_dir: str) -> List[Document]:
        """Iterates through docs, creates chunks, and assigns metadata."""
        all_final_chunks = []

        for root, _, files in os.walk(raw_docs_dir):
            for file in files:
                if not file.endswith(('.html', '.md', '.txt')):
                    continue

                file_path = os.path.join(root, file)
                source_domain = "python_docs" if "python_docs" in file_path else "langchain_docs"

                # Extract basic structural metadata
                base_metadata = {
                    "source": source_domain,
                    "filename": file,
                    "document_id": file_path,
                    "url": f"https://{source_domain}/{file}"  # Placeholder logic
                }

                # 1. Branching logic based on file type
                if file.endswith('.html'):
                    # Returns pre-chunked Document objects with HTML metadata
                    html_chunks = self.process_html_file(file_path)

                    # Merge base metadata into the structurally extracted metadata
                    for chunk in html_chunks:
                        chunk.metadata.update(base_metadata)
                        all_final_chunks.append(chunk)

                else:
                    # For non-HTML files, read as single string
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()

                    # Create a single document
                    doc = Document(page_content=text_content, metadata=base_metadata)

                    # Run it through the standard recursive splitter
                    text_chunks = self.text_splitter.split_documents([doc])
                    all_final_chunks.extend(text_chunks)

        # 2. Add chunk-level metadata
        for i, chunk in enumerate(all_final_chunks):
            chunk.metadata["chunk_index"] = i

        return all_final_chunks