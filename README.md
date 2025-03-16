# sql_rag_agent_local_llm

This project utilizes the Ollama language model to perform SQL retrieval and RAG (Retrieval-Augmented Generation) tasks. The following sections provide instructions on how to set up and use the project.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Chrisliao0806/sql_rag_agent_local_llm.git
    cd sql_rag_agent_local_llm
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add the following:
    ```env
    TAVILY_API_KEY=your_tavily_api_key
    ```

## Usage

1. Run the main script with the desired arguments:
    ```sh
    python main.py --question "你的問題" --db-uri "sqlite:///data/Innodisk.db" --pdf-file "data/grok_file.pdf" --chroma-file "data/chroma_db" --model "qwen2.5:7b"
    ```

2. Available command-line arguments:
    - `--log-level`: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".
    - `--question`: The question to ask the model. Default is "可以幫我從Issue_Header找出最常見的issue嗎？".
    - `--db-uri`: The URI of the database. Default is "sqlite:///data/Innodisk.db".
    - `--pdf-file`: The path to the PDF file. Default is "data/grok_file.pdf".
    - `--chroma-file`: The path to the chroma database. Default is "data/chroma_db".
    - `--model`: The name of the model to use for embedding. Default is "qwen2.5:7b".
    - `--chunk-size`: The maximum size of each text chunk. Default is 300.
    - `--chunk-overlap`: The number of characters that overlap between chunks. Default is 10.

## Project Structure

- `main.py`: The entry point of the application. Parses command-line arguments and initializes the retrieval workflow.
- `rag_sql_chain.py`: Contains the `RetrieveBot` class, which handles the SQL retrieval and RAG tasks.
- `utils/logger.py`: Utility for setting up logging.
- `utils/prompt.py`: Contains prompt templates for different tasks.
- `utils/choose_state.py`: Defines the state classes used in the retrieval process.

## Dependencies

- `argparse`
- `logging`
- `dotenv`
- `langchain`
- `langchain_community`
- `langchain_core`
- `langchain_ollama`
- `tavily_search`
- `chroma`
- `PyMuPDF`
- `HuggingFaceEmbeddings`

Make sure to install all dependencies listed in `requirements.txt`.

## License

This project is licensed under the MIT License.