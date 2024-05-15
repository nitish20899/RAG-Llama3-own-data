import chainlit as cl
import shutil
from pathlib import Path
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import textwrap
import os

UPLOAD_DIR = Path("./data")
os.environ["GROQ_API_KEY"] = <GROQ_API_KEY>
Llamaparse_api_key = <LLAMAPARSE_API_KEY>

instruction = """The provided document may contain various types of information, such as financial reports, research papers, technical manuals, or other text content.
Your task is to parse the document and answer questions based on the content.
Be precise and refer to the document for accurate responses."""

parser = LlamaParse(
    api_key=Llamaparse_api_key,
    result_type="markdown",
    parsing_instruction=instruction,
    max_timeout=5000,
)

@cl.on_chat_start
async def on_chat_start():
    files = None  # Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,  # Optionally limit the file size
            timeout=180,  # Set a timeout for user response
        ).send()

    file = files[0]  # Get the first uploaded file
    print(file)  # Print the file object for debugging

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Process the PDF file using the path
    await process_pdf(file.path, msg)

async def process_pdf(file_path, msg):
    # Initialize global variables
    global qa, is_ready
    is_ready = False

    try:
        # Parse the PDF
        llama_parse_documents = await parser.aload_data(file_path)

        parsed_doc = llama_parse_documents[0]
        document_path = UPLOAD_DIR / "parsed_document.md"
        with document_path.open("w") as f:
            f.write(parsed_doc.text)

        loader = UnstructuredMarkdownLoader(document_path)
        loaded_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
        docs = text_splitter.split_documents(loaded_documents)

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        qdrant = await cl.make_async(Qdrant.from_documents)(
            docs,
            embeddings,
            path="./db_new",
            collection_name="document_embeddings",
        )

        retriever = qdrant.as_retriever(search_kwargs={"k": 5})

        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Answer the question and provide additional helpful information,
        based on the pieces of information, if applicable. Be succinct.

        Responses should be properly formatted to be easily read.
        """

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Initialize message history for conversation
        message_history = ChatMessageHistory()

        # Memory for conversational context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create a ConversationalRetrievalChain that uses the Qdrant vector store and memory
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatGroq(temperature=0, model_name="llama3-70b-8192"),
            retriever=compression_retriever,
            memory=memory,
            return_source_documents=True,
        )

        # Inform the user that processing is done
        msg.content = f"Processing `{file_path}` done. You can now ask questions!"
        await msg.update()
        is_ready = True

    except Exception as e:
        is_ready = False
        await msg.update(f"Error processing PDF: {e}")

@cl.on_message
async def main(message: cl.Message):
    global qa, is_ready
    if not is_ready:
        await cl.Message(content="The QA system is not ready. Please upload a PDF first.").send()
        return

    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    res = await qa.ainvoke(message.content, callbacks=[cb])
    print("Response:", res)  # Print the response for debugging

    # Extract answer and source_documents safely
    answer = res.get("answer", "No answer found.")
    source_documents = res.get("source_documents", [])

    text_elements = []  # Initialize list to store text elements

    # # Process source documents if available
    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )

    # Return results
    await cl.Message(content=answer, elements=text_elements).send()

if __name__ == "__main__":
    cl.App().run(port=8500)
