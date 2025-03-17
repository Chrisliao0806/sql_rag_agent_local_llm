import logging
import os
import re

from dotenv import load_dotenv
from langchain import hub
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from utils.choose_state import (
    GradeDocuments,
    PlainState,
    QueryOutput,
    RAGState,
    State,
    WebState,
)
from utils.prompt import (
    INSTRUCTIONPLAIN,
    INSTRUCTIONRAG,
    INSTRUCTIONRAGGRADE,
    INSTRUCTIONWEB,
    INSTRUCTIONWEBRAG,
    SQLTEMPLATE,
)

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class RetrieveBot:
    """
    Attributes:
        llm (ChatOllama): The language model used for generating SQL queries and answers.
        query_prompt_template (PromptTemplate): The template for generating SQL query prompts.
        db (SQLDatabase): The database connection object.

    Methods:
        __init__(db_uri, model="qwen2.5:7b"):
            Initializes the SqlRetrieve object with the given database URI and model.

        _db_query(query="SELECT Subject FROM Issue_Header GROUP BY Subject ORDER BY COUNT(*) DESC LIMIT 25"):
            Executes a database query and prints the results.

        _show_prompt():

        write_query(state: State) -> dict:

        execute_query(state: State) -> dict:
            Executes the SQL query stored in the state and returns the result.

        generate_answer(state: State) -> dict:
            Generates an answer to the user's question using the retrieved information as context.

        workflow(query: str):
            Defines the workflow for the SQL retrieval process, including writing the query,
            executing the query, and generating the answer.
    """

    def __init__(
        self,
        db_uri,
        chroma_file,
        pdf_file="data/grok_file.pdf",
        chunk_size=300,
        chunk_overlap=10,
        model="qwen2.5:7b",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_file = chroma_file
        self.pdf_reader = PyMuPDFLoader(pdf_file).load()
        self.document_embedding()
        self.llm = ChatOllama(model=model, base_url="http://localhost:11434")
        self.query_prompt_template = PromptTemplate.from_template(SQLTEMPLATE)
        self.db = SQLDatabase.from_uri(db_uri)
        (
            self.rag_chain,
            self.llm_chain,
            self.web_chain,
            self.question_router,
            self.retrieval_grader,
        ) = self._init_model()
        self.web_search_tool = TavilySearchResults(
            include_answer=True,
        )

    def _init_model(self):
        """
        Initializes the language model used for generating SQL queries and answers.
        """
        prompt_rag = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONRAG),
                ("system", "文件: \n\n {documents}"),
                ("human", "問題: {question}"),
            ]
        )

        prompt_plain = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONPLAIN),
                ("human", "問題: {question}"),
            ]
        )

        prompt_web = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONWEBRAG),
                ("human", "問題: {question}"),
                ("system", "網頁內容: \n\n {documents}"),
            ]
        )
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONWEB),
                ("human", "問題: {question}"),
            ]
        )
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONRAGGRADE),
                ("human", "文件: \n\n{documents}"),
                ("human", "問題: {question}"),
            ]
        )

        # LLM & chain
        rag_chain = prompt_rag | self.llm | StrOutputParser()
        # LLM & chain
        llm_chain = prompt_plain | self.llm | StrOutputParser()
        # LLM & chain
        web_chain = prompt_web | self.llm | StrOutputParser()
        # Route LLM with tools use
        structured_llm_router = self.llm.bind_tools(tools=[WebState, PlainState])
        question_router = route_prompt | structured_llm_router

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        # 使用 LCEL 語法建立 chain
        retrieval_grader = grade_prompt | structured_llm_grader
        return (
            rag_chain,
            llm_chain,
            web_chain,
            question_router,
            retrieval_grader,
        )

    def _db_query(self, query: str):
        """
        Executes a database query and prints the results.
        """
        print(self.db.dialect)
        print(self.db.get_usable_table_names())
        print(self.db.run(query))

    def _show_prompt(self):
        """
        Displays the prompt message in a formatted manner.

        This method asserts that there is exactly one message in the
        query_prompt_template's messages list and then prints it
        using the pretty_print method.
        """
        assert len(self.query_prompt_template.messages) == 1
        self.query_prompt_template.messages[0].pretty_print()
        

    def _clean_sql_string(self,sql_string):
        sql_string = re.sub(r'```sql|```', '', sql_string)
        sql_string = sql_string.replace('\\n', ' ')
        sql_string = re.sub(r'\s+', ' ', sql_string)
        return sql_string.strip()

    def document_embedding(self):
        """
        Generates document embeddings and initializes a vector database for retrieval.

        This method performs the following steps:
        1. Embeds the text using a HuggingFace model.
        2. Splits the text into chunks using a RecursiveCharacterTextSplitter.
        3. Creates a Chroma vector database from the split documents.
        4. Initializes a retriever from the vector database for document retrieval.

        Attributes:
            model_name (str): The name of the HuggingFace model to use for embedding.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between text chunks.
            pdf_reader (object): The PDF reader object containing the documents to be split.
            vectordb (Chroma): The Chroma vector database created from the documents.
            retriever (object): The retriever initialized from the vector database.

        Returns:
            None
        """
        # Embed text
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "mps"},
        )
        if os.path.exists(self.chroma_file):
            self.vectordb = Chroma(
                persist_directory=self.chroma_file,
                embedding_function=embedding,
                collection_name="coll2",
                collection_metadata={"hnsw:space": "cosine"},
            )
            logging.info("Chroma file exists")
            print("Chroma file exists")
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            logging.info("Text splitting completed")

            all_splits = text_splitter.split_documents(self.pdf_reader)
            self.vectordb = Chroma.from_documents(
                documents=all_splits,
                embedding=embedding,
                collection_name="coll2",
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=self.chroma_file,
            )
            self.vectordb.persist()
            logging.info("Chroma file created")
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, state):
        """
        Retrieve relevant documents based on the given question in the state.

        Args:
            state (dict): A dictionary containing the question to be used for retrieval.

        Returns:
            dict: A dictionary containing:
                - "documents" (list): A list of tuples where each tuple
                                      contains a document and its relevance score.
                - "question" (str): The original question from the state.
                - "use_rag" (bool): A flag indicating whether the relevance score
                                    of any document exceeds the threshold (0.3).
        """

        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        # 0.3 is the threshold for relevance score
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def retrieval_grade(self, state):
        """
        filter retrieved documents based on question.

        Args:
            state (dict):  The current state graph

        Returns:
            state (dict): New key added to state, documents, that contains list of related documents.
        """

        # Grade documents
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        documents = state["documents"]
        question = state["question"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"documents": d.page_content, "question": question}
            )
            grade = score.binary_score
            if grade == "yes":
                print("  -GRADE: DOCUMENT RELEVANT-")
                filtered_docs.append(d)
            else:
                print("  -GRADE: DOCUMENT NOT RELEVANT-")
                continue
        return {"documents": filtered_docs, "question": question}

    def rag_generate(self, state):
        """
        Generates a response in RAG (Retrieval-Augmented Generation) mode.

        This method takes a state dictionary containing a question and a list of documents,
        and uses the RAG chain to generate a response based on the provided documents and question.

        Args:
            state (dict): A dictionary containing the following keys:
                - "question" (str): The question to be answered.
                - "documents" (list): A list of documents to be used for generating the response.

        Returns:
            dict: A dictionary containing the original question,
                  documents, and the generated response.
                - "question" (str): The original question.
                - "documents" (list): The original list of documents.
                - "generation" (str): The generated response.
        """
        print("---GENERATE IN RAG MODE---")
        question = state["question"]
        documents = state["documents"]
        # RAG generation
        generation = self.rag_chain.invoke(
            {"documents": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def first_stage_end(self, state):
        """
        End of the first stage.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        print("---FIRST STAGE END---")
        return {"question": state["question"]}

    def route_retrieval(self, state):
        """
        Determines whether to generate an answer, or use websearch.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ROUTE RETRIEVAL---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            print(
                "  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO PLAIN ANSWER-"
            )
            return "plain_feedback"
        else:
            # We have relevant documents, so generate answer
            print("  -DECISION: GENERATE WITH RAG LLM-")
            return "rag_generate"

    ### Edges ###
    def route_web_test(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        logging.info("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})

        if len(source.tool_calls) == 0:
            logging.info("  -ROUTE TO PLAIN LLM-")
            return "plain_feedback"

        if source.tool_calls[0]["name"] == "WebState":
            logging.info("  -ROUTE TO WEB SEARCH-")
            return "web_search"

        elif source.tool_calls[0]["name"] == "PlainState":
            logging.info("  -ROUTE TO PLAIN LLM-")
            return "plain_feedback"

    def web_generate(self, state):
        """
        Generates a response using web search.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains web search generation
        """
        logging.info("---WEB GENERATE---")
        question = state["question"]
        documents = self.web_search_tool.invoke({"query": question})
        documents = [doc["content"] for doc in documents]
        # RAG generation
        generation = self.web_chain.invoke(
            {"documents": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def plain_answer(self, state):
        """
        Generate answer using the LLM without vectorstore.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        print("---GENERATE PLAIN ANSWER---")
        question = state["question"]
        generation = self.llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}

    def route_sql_test(self, state):
        """
        Route question to SQL or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")

        if state["result"] == "查無結果":
            print("  -ROUTE TO RAG -")
            return "rag_generate"

        else:
            print("  -ROUTE TO SQL -")
            return "sql_feedback"

    def write_query(self, state: State):
        """
        Generates and executes a query based on the provided state.

        Args:
            state (State): The current state containing the question to be converted into a query.

        Returns:
            dict: A dictionary containing the generated query under the key "query".
        """
        try:
            prompt = self.query_prompt_template.invoke(
                {
                    "dialect": self.db.dialect,
                    "top_k": 25,
                    "table_info": self.db.get_table_info(),
                    "input": state["question"],
                }
            )
            result = self.llm.invoke(prompt)
            print(result)
            if result is None:
                return {"query": "查無結果"}
            result = self._clean_sql_string(result.content)
            return {"query": result}
        except Exception as e:
            logging.error("Error generating query: %s", e)
            return {"query": "查無結果"}

    def execute_query(self, state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        result = execute_query_tool.invoke(state["query"])
        if "Error" in result or result is None or result == "":
            return {"result": "查無結果"}
        return {"result": result}

    def generate_answer(self, state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}"
        )
        response = self.llm.invoke(prompt)
        return {"generation": response.content}

    def workflow(self, query: str):
        """
        Define the workflow for the SQL retrieval process.

        This method constructs a state graph for the SQL retrieval process,
        which includes writing the query, executing the query, and generating
        the answer. It then streams the steps of the graph while printing
        the progress and token usage.

        Args:
            query (str): The SQL query to be processed.

        Returns:
            None
        """
        token = []
        workflow = StateGraph(State)
        workflow.add_node("write_query", self.write_query)  # write query
        workflow.add_node("execute_query", self.execute_query)  # execute query
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("rag_generate", self.rag_generate)  # generate in RAG mode
        workflow.add_node("sql_feedback", self.generate_answer)  # generate answer
        workflow.add_node("first_stage_end", self.first_stage_end)  # end of first stage
        workflow.add_node("web_search", self.web_generate)  # web search
        workflow.add_node("plain_feedback", self.plain_answer)  # plain answer
        workflow.add_node("grade_documents", self.retrieval_grade)  # grade documents

        workflow.add_edge(START, "write_query")
        workflow.add_edge("write_query", "execute_query")
        workflow.add_conditional_edges(
            "execute_query",
            self.route_sql_test,
            {
                "rag_generate": "retrieve",
                "sql_feedback": "sql_feedback",
            },
        )
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.route_retrieval,
            {
                "rag_generate": "rag_generate",
                "plain_feedback": "first_stage_end",
            },
        )
        workflow.add_conditional_edges(
            "first_stage_end",
            self.route_web_test,
            {
                "web_search": "web_search",
                "plain_feedback": "plain_feedback",
            },
        )
        workflow.add_edge("rag_generate", END)
        workflow.add_edge("sql_feedback", END)
        workflow.add_edge("web_search", END)
        workflow.add_edge("plain_feedback", END)

        compiled_app = workflow.compile()
        with get_openai_callback() as cb:
            output = compiled_app.invoke({"question": query})
            token.append(cb.total_tokens)
            token.append(cb.prompt_tokens)
            token.append(cb.completion_tokens)

        image_data = compiled_app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
        FILE = "mermaid_diagram.png"
        try:
            with open(FILE, "wb") as f:
                f.write(image_data)
            print(f"image save: {FILE}")
        except IOError as e:
            print(f"image save error: {e}")
        return output["generation"], token
