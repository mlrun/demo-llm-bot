import os
from typing import List

import fsspec
import openai
from langchain.agents import AgentExecutor, Tool, initialize_agent, load_tools
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTListIndex,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    download_loader,
    load_index_from_storage,
)
from llama_index.indices.base import BaseGPTIndex
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.readers.schema.base import Document

INDEX_KINDS = {
    "list": GPTListIndex,
    "vector": GPTVectorStoreIndex,
}
DEFAULT_INDEX_CLASS = GPTVectorStoreIndex


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")


class LlmIndex:
    def __init__(self, persist_dir: str = "./storage", additional_tools: list = None):
        """Class wrapper for llama_index

        Usage example::

            # loading documents, creating an index and saving it all to storage
            llmi = LlmIndex(additional_tools=["llm-math"])
            llmi.add_index(
                index_name="llama",
                documents=llmi.load_documents(documents_dir="data/text"),
                summary="useful for questions about llamas"
            )
            llmi.persist()

            # load the docs & indexes from storage and run queries
            llmi = LlmIndex(additional_tools=["llm-math"])
            llmi.load_storage()
            print(llmi.query("What is 5 times the average weight of llamas?"))

        :param persist_dir: default storage files location
        :param additional_tools: List of additional tools for the langchain agent to use
        """

        # Llama store setup
        self.persist_dir = persist_dir
        self.indices = {}
        self._llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self._llm_predictor = LLMPredictor(llm=self._llm)
        self._storage_context = StorageContext.from_defaults()
        self._service_context = ServiceContext.from_defaults(llm_predictor=self._llm_predictor)
        # self._query_engines = {}

        # Langchain agent tools
        self._additional_tools = additional_tools if additional_tools else []
        self._tools = []

        # Langchain agent setup
        self._memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent_executor = None

    @staticmethod
    def _dir_to_fsspec(persist_dir):
        if "://" in persist_dir:
            import mlrun

            store, persist_dir = mlrun.store_manager.get_or_create_store(persist_dir)
            fs = store.get_filesystem(False)
        else:
            fs = fsspec.filesystem("file")
        return fs, persist_dir

    def load_storage(self, persist_dir=None, load_indices=True, index_args: dict = None):
        """Load objects (indexes, documents, vectors) from the storage context

        :param persist_dir:  storage files location (if not specified it will use the one set in init)
        :param load_indices: indicate id indexes should be loaded into memory
        """
        fs, persist_dir = self._dir_to_fsspec(persist_dir or self.persist_dir)
        self._storage_context = StorageContext.from_defaults(persist_dir=persist_dir, fs=fs)

        if load_indices:
            for key, summary in self.get_catalog().items():
                args = index_args.get(key, {}) if index_args else {}
                self.load_index(key, index_args=args)
                print(f"Loaded {key} index {summary}")

    def persist(self, persist_dir=None):
        """Persist objects (indexes, documents, vectors) to the storage context

        :param persist_dir: storage files location (if not specified it will use the one set in init)
        """
        fs, persist_dir = self._dir_to_fsspec(persist_dir or self.persist_dir)
        self._storage_context.persist(persist_dir=persist_dir, fs=fs)

    @staticmethod
    def load_documents(loader=None, documents_dir="", **kwargs) -> List[Document]:
        """Load a document from a dir or using a loader class

        :param loader:         document loader class name
        :param documents_dir:  directory to load files from (using SimpleDirectoryReader)
        :param kwargs:         loader class arguments
        :return: list of document objects
        """
        if loader:
            loader_class = download_loader(loader)
            documents = loader_class().load_data(**kwargs)
        else:
            documents = SimpleDirectoryReader(documents_dir).load_data()
        return documents

    def _check_index_name(self, index_name):
        if index_name not in self.indices:
            raise ValueError(
                f"Unknown index {index_name}, loaded indices: {'|'.join(self.indices.keys())}"
            )

    def add_index(
        self, index_name, documents, summary="", index_class=None, **kwargs
    ) -> BaseGPTIndex:
        """Add a new query index and add documents to it

        :param index_name:  name of the index
        :param documents:   list of document objects
        :param summary:     summary text for the index content
        :param index_class: Index class (e.g. GPTListIndex, GPTVectorStoreIndex)
        :return: index object
        """
        index_class = index_class or DEFAULT_INDEX_CLASS
        index = index_class.from_documents(
            documents,
            service_context=self._service_context,
            storage_context=self._storage_context,
            **kwargs,
        )
        index.set_index_id(index_name)
        if summary:
            index.summary = summary
        self.indices[index_name] = index
        return index

    def add_tool(self, tool: Tool) -> None:
        self._tools.append(tool)

    def add_docs_to_index(self, index_name, documents):
        """Add documents to existing index

        :param index_name: name of the index
        :param documents:  list of document objects
        """
        self._check_index_name(index_name)
        for doc in documents:
            self.indices[index_name].insert(doc)

    def load_index(self, index_name: str, index_args: dict) -> BaseGPTIndex:
        """Load an index into memory

        :param index_name: name of the index
        :return: index object
        """
        index = load_index_from_storage(self._storage_context, index_id=index_name, **index_args)
        self.indices[index_name] = index
        return index

    def get_catalog(self) -> dict:
        catalog = {}
        indices = self._storage_context.index_store.index_structs()
        for index in indices:
            catalog[index.index_id] = index.summary
        return catalog

    def _get_or_create_agent_executor(self, refresh=False) -> AgentExecutor:
        if refresh or not self.agent_executor:
            self.agent_executor = initialize_agent(
                tools=self._load_all_tools(),
                llm=self._llm,
                agent="conversational-react-description",
                memory=self._memory,
                verbose=True,
            )
        return self.agent_executor

    def _load_indicies_as_tools(self, index_args=None) -> List[Tool]:
        indicies_as_tools = []
        for key, summary in self.get_catalog().items():
            args = index_args.get(key, {}) if index_args else {}
            index = self.load_index(key, index_args=args)
            index_tool = LlamaIndexTool.from_tool_config(
                IndexToolConfig(
                    query_engine=index.as_query_engine(),
                    name=key,
                    description=summary,
                    # tool_kwargs={"return_direct": True, "return_sources": True},
                )
            )
            indicies_as_tools.append(index_tool)
        return indicies_as_tools

    def _load_additional_tools(self) -> List[Tool]:
        return load_tools(tool_names=self._additional_tools, llm=self._llm)

    def _load_all_tools(self) -> List[Tool]:
        return self._tools + self._load_additional_tools() + self._load_indicies_as_tools()

    def query(self, query_str: str) -> str:
        """Query the index with a text question

        :param query_str:  question in natural language
        :return: text response
        """
        agent_executor = self._get_or_create_agent_executor()
        return agent_executor.run(query_str)


class QueryLlamaGpt:
    def __init__(self, persist_dir: str = "./storage", additional_tools: list = None):
        self.llama = LlmIndex(
            persist_dir=persist_dir, additional_tools=additional_tools if additional_tools else []
        )
        self.llama.load_storage()

    def do(self, event):
        event.body["response"] = self.llama.query(event.body["query"])
        return event
