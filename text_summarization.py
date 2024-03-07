import os
import langchain
import openai
import warnings
import textwrap
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# fro storing word vectors
from langchain.vectorstores import FAISS
# for getting the documents
from langchain.chains import RetrievalQA
# for loading the document
from langchain.document_loaders import TextLoader
# converting our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import pandas as pd
import json
warnings.filterwarnings('ignore')
os.environ['OPENAI_API_KEY'] = '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'
key = os.environ.get('OPENAI_API_KEY')


# creating Open AI object class
class OpenAIObject:

    def __init__(self, model=None):

        if model is not None:
            self.llm = OpenAI(temperature=0, openai_api_key=key, model=model)
        else:
            self.llm = OpenAI(temperature=0, openai_api_key=key)


class ChatModelAPI:

    def __init__(self):
        self.chat_model = ChatOpenAI(
            temperature=0, openai_api_key=key, max_tokens=1000)
        

class LoadFileAsDocument:

    @classmethod
    def load_file_as_doc(cls,file_name):

        try:
            loader = TextLoader(file_name, encoding='unicode_escape')
            return loader.load()
        except Exception as e:
            return e


class SummarizeChains:

    @classmethod
    def text_summarise(cls,model,chain,prompt):
        try:
            return load_summarize_chain(
                llm=model, chain_type=chain, verbose=True, prompt=prompt)
        except Exception as e:
            return e

class TextSplitter:

    @classmethod
    def split_character(cls, seprator=None, chunksize=None, chunkoverlap=None):

        try:
            return RecursiveCharacterTextSplitter(seprator, chunksize, chunkoverlap)
        except Exception as e:
            return e


class QARetrival:

    @classmethod
    def retrivalqa(cls, chain, docs, embeddings, model):
        try:
            docsearch = FAISS.from_documents(
                docs, embeddings)
            return RetrievalQA.from_chain_type(
                llm=model, chain_type=chain, retriever=docsearch.as_retriever())
        except Exception as e:
            return e
# creating summarisation class for
# creating summary for any given text
class TextSummarisation(OpenAIObject):

    def __init__(self, model=None):
        super().__init__(model)

    # function for cretaing instruction template
    # this function is used to pass instruction
    # to GPT model for displaying the output
    def command_instructions(self):

        try:
            template = '''
             %INSTRUCTIONS:
             please summarize the following piece of text.
             respond in a manner that a 5 years old can understand.
             
             
             %TEXT:
             {text}
             '''
            return PromptTemplate(input_variables=["text"], template=template)

        except Exception as e:
            pass

    # function to create summary of any given context
    # generates the answers as per the instructions given before
    def summarise_text(self, user_input):

        try:
            prompt = self.command_instructions()
            final_prompt = prompt.format(text=user_input)
            return self.llm(final_prompt)
        except Exception as e:
            return e




# class for generating summary for long text
class LongTextSummarisation(OpenAIObject):

    def __init__(self, model=None):

        super().__init__(model=model)

    # creating documents for the large text inputs
    # passing the input as batch to the model
    def generate_docs(self, delimeter=None, size=None, overlap=None, text=None):

        try:

            return TextSplitter().split_character(seprator=delimeter, chunksize=size, chunkoverlap=overlap).create_documents([text])
        except Exception as e:
            return e

    def instruction_prompt(self):

        try:
            template = '''
             %INSTRUCTIONS:
             write a conscise point wise summary of the following.
             display points as hard bullet points.
             apply fullstop after every point.
             
             %TEXT:
             {text}
             '''
            return PromptTemplate(input_variables=["text"],    template=template)

        except Exception as e:
            pass

    # summarizing each batch input
    def summarise_long_text(self, delimeter=None, size=None, overlap=None,user_input=None):

        try:
            chain = SummarizeChains().text_summarise(model=self.llm,chain='stuff',prompt=self.instruction_prompt())
            return chain.run(self.generate_docs(delimeter, size, overlap,user_input))
        except Exception as e:
            return e


# class for Generating Summary for an
# entire document
'''
in this following code, i have used FAISS vectorstore with embaddings and RetrivelQA to get summary of the document.
since load_summarize_chain with gpt3.5 and gpt3.5-turbo give problem with large documents
class DocumentSummarisation(OpenAIObject):

    def __init__(self, model=None, filepath=None):

        super().__init__(model=model)
        self.file = filepath

    # loading the text file
    def load_file(self):

        try:
            text = open(self.file, 'r', encoding='unicode_escape')
            return text.read()
        except Exception as e:
            return e

    # generating docs for each line
    # of the file
    def generating_docs(self):

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=300)
            return text_splitter.create_documents([self.load_file()])
        except Exception as e:
            return e

    # summarizing the docs and generating
    # overall summary
    def summarise_doc_text(self):

        try:
            chain = load_summarize_chain(llm=self.llm, chain_type='map_reduce')
            return chain.run(self.generating_docs())
        except Exception as e:
            return e


'''
class DocumentSummarisation(OpenAIObject):

    def __init__(self, model=None, filepath=None):

        super().__init__(model=model)
        self.file = filepath

    # loading the text file

    def load_file(self):

        try:
            return LoadFileAsDocument().load_file_as_doc(file_name=self.file)
        except Exception as e:
            return e

    # generating docs for each line
    # of the file
    def generating_docs(self, delimeter=None, size=None, overlap=None):

        try:
            return TextSplitter().split_character(seprator=delimeter, chunksize=size, chunkoverlap=overlap).split_documents(self.load_file())
        except Exception as e:
            return e

    def doc_instruction_prompt(self):

        try:
            template = ''' %INSTRUCTIONS:
             write a conscise 10 points summary of the following.
             display points as bullets insted of numbers.
             display bullet points as arrows.
             apply fullstop after every point.
             try to summarize each content in the document.
             

             %TEXT:
             {text}
             
             CONSCISE SUMMARY:
             '''
            return PromptTemplate(input_variables=["text"], template=template)
        except Exception as e:
            return e

    def summary_template_design(self, query):
        try:
            prompt = self.doc_instruction_prompt()
            return prompt.format(text=query)
        except Exception as e:
            return e

    def get_embeddings(self):
        try:
            return OpenAIEmbeddings(openai_api_key=key)
        except Exception as e:
            return e

    # summarizing the docs and generating
    # overall summary
    def summarise_doc_text(self, delimeter=None, size=None, overlap=None, user_query=None):
        try:
            return (QARetrival.retrivalqa(chain='stuff', docs=self.generating_docs(delimeter, size, overlap), embeddings=self.get_embeddings(), model=self.llm).run(self.summary_template_design(query=user_query)))
        except Exception as e:
            return e


# class for Questning Answering
# for the given document
class DocumentQA(OpenAIObject):

    def __init__(self, model=None, filepath=None):
        super().__init__(model)
        self.file = filepath

    # function for loading file

    def load_file(self):

        try:
            return LoadFileAsDocument().load_file_as_doc(file_name=self.file)
        except Exception as e:
            return e

    # function for creating batch inputs
    def create_batch(self, delimeter=None, size=None, overlap=None):

        try:
            return TextSplitter().split_character(seprator=delimeter, chunksize=size, chunkoverlap=overlap).split_documents(self.load_file())
        except Exception as e:
            return e

    # function for generating embaddings
    def get_embeddings(self):
        try:
            return OpenAIEmbeddings(openai_api_key=key)
        except Exception as e:
            return e

    def command_qa_instructions(self):
        try:
            template = '''
              %INSTRUCTIONS:
              Answer the following question only based on the document content.
              also if any persons name or service name occurs,first search the name in the document and then provide the answer with respect to document content.
              check if the prompt is related to the document.
              if any given text from the prompt is not recognized, or not found in the document simply reply as "sorry!!, may be the thing you are looking for is not found in the document".
              
              
          
              %TEXT:
              {text}
              '''
            return PromptTemplate(input_variables=["text"],    template=template)
        except Exception as e:
            pass

    def prompt_template_design(self, query):
        try:
            prompt = self.command_qa_instructions()
            return prompt.format(text=query)
        except Exception as e:
            return e

    def run_engine(self, delimeter=None, size=None, overlap=None, query=None):
        try:

            return (QARetrival.retrivalqa(chain='stuff', docs=self.create_batch(delimeter, size, overlap), embeddings=self.get_embeddings(), model=self.llm).run(self.prompt_template_design(query=query)))
        except Exception as e:
            return e


# class for predicting context for
# a given line in the file
class ContextMapping(ChatModelAPI):

    def __init__(self, filepath='E:\MoneyControl Data Scraping\LinkedInIndustries.csv'):
        super().__init__()
        self.df = pd.read_csv(filepath)

    # generating mappings structure
    # for input and output

    def get_mapping_structure(self):

        try:
            response_schemas = [
                ResponseSchema(
                    name="input_industry", description="This is the input_industry from the user"),
                ResponseSchema(name="standardized_industry",
                               description="This is the industry you feel is most closely matched to the users input"),
                ResponseSchema(
                    name="match_score",  description="A score 0-100 of how close you think the match is between user input and your match")
            ]
            return StructuredOutputParser.from_response_schemas(response_schemas)
        except Exception as e:
            return e

    # parsing the structure
    def parsing_output_structure(self):
        try:
            return self.get_mapping_structure().get_format_instructions()
        except Exception as e:
            return e

    # this tells chat gpt what input
    # is given and how to generate the output
    def generating_prompt_structure(self):

        try:
            template = """
                      You will be given a series of industry names from a user.
                      Find the best corresponding match on the list of standardized names.
                      The closest match will be the one with the closest semantic meaning. Not just string similarity.
                      
                      {format_instructions}
                      
                      Wrap your final output with closed and open brackets (a list of json objects)
                      
                      input_industry INPUT:
                      {user_industries}
                      
                      STANDARDIZED INDUSTRIES:
                      {standardized_industries}
                      
                      YOUR RESPONSE:
                      """

            return ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(template)
                ],
                input_variables=["user_industries", "standardized_industries"],
                partial_variables={
                    "format_instructions": self.parsing_output_structure()}
            )
        except Exception as e:
            return e

    # generating the context
    def get_context(self, userinput):

        try:
            _input = self.generating_prompt_structure().format_prompt(user_industries=userinput,
                                                                      standardized_industries=", ".join(self.df['Industry'].values))
            return self.chat_model(_input.to_messages()).content

        except Exception as e:
            return e


if __name__ == '__main__':
    pass
