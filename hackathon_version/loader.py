from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def load_domain_data(directory="./pdf"):
    documents = SimpleDirectoryReader(directory).load_data()
    splitter = SentenceSplitter(chunk_size=350, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    chunks = [{"text": n.get_content(), "source": n.metadata.get("file_name", "unknown")} for n in nodes]
    return chunks
