import argparse
import logging
from utils.logger import setup_logging
from rag_sql_chain import RetrieveBot


def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: The parsed arguments.

    The following arguments are supported:
        --log-level (str): The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Default is "INFO".
        --question (str): The question to ask the model. Default is "可以幫我找出最常見的issue problem嗎？".
        --db-uri (str): The URI of the database. Default is "sqlite:///data/Innodisk.db".
        --pdf-file (str): The path to the PDF file. Default is "data/grok_file.pdf".
        --model (str): The name of the model to use for embedding. Default is "qwen2.5:7b".
        --chunk-size (int): The maximum size of each text chunk. Default is 300.
        --chunk-overlap (int): The number of characters that overlap between chunks. 
            Default is 10.
    """
    parser = argparse.ArgumentParser(description="rag and sql chain model for using langgraph")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        help="The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--question",
        default="可以幫我從Issue_Header找出最常見的issue嗎？",
        type=str,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--db-uri",
        default="sqlite:///data/Innodisk.db",
        type=str,
        help="The URI of the database.",
    )
    parser.add_argument(
        "--pdf-file",
        default="data/grok_file.pdf",
        type=str,
        help="The path to the PDF file.",
    )
    parser.add_argument(
        "--chroma-file",
        default="data/chroma_db",
        type=str,
        help="The path to the chroma database.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        type=str,
        help="The name of the model to use for embedding.",
    )
    parser.add_argument(
        "--chunk-size",
        default=300,
        type=int,
        help="The maximum size of each text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        default=10,
        type=int,
        help="The number of characters that overlap between chunks.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(log_level=args.log_level)
    logging.info("Parsed command-line arguments")
    sql_retrieve = RetrieveBot(
        db_uri=args.db_uri,
        pdf_file=args.pdf_file,
        chroma_file=args.chroma_file,
        model=args.model,
    )
    output_answer, token = sql_retrieve.workflow(query=args.question)
    logging.info("Answer: %s", output_answer)
    logging.info("Token: %s", token)
    logging.info("Completed")
