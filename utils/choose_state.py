from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from typing_extensions import Annotated
from typing import List


class State(TypedDict):
    """State for the SQL retrieval process
    question: The question to ask the model.
    query: The generated SQL query.
    result: The result of the SQL query.
    answer: The answer to the question.
    """

    question: str
    documents: List[str]
    query: str
    result: str
    generation: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


class RAGState(BaseModel):
    """
    向量資料庫回覆工具。若問題可以從向量資料庫中找到答案，則使用RAG工具回覆。
    """

    query: str = Field(description="使用向量資料庫回覆時輸入的問題")


class PlainState(BaseModel):
    """
    直接回覆工具。若問題從向量資料庫中找不到的話，則直接用自己的知識進行回覆
    """

    query: str = Field(description="使用直接回覆時輸入的問題")


class SqlState(BaseModel):
    """
    資料庫回覆工具。若問題可以從資料庫中找到答案，則直接用Sql工具進行回覆
    """

    query: str = Field(description="使用資料庫回覆時輸入的問題")


class WebState(BaseModel):
    """
    網路搜尋工具。若問題覺得需要用網路查詢，則使用WebState工具搜尋解答。
    """

    query: str = Field(description="使用網路搜尋時輸入的問題")


class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")
