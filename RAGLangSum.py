#%%capture
#!pip install "ibm-watsonx-ai==0.2.6"
#!pip install "langchain==0.1.16" 
#!pip install "langchain-ibm==0.1.4"
#!pip install "huggingface == 0.0.1"
#!pip install "huggingface-hub == 0.23.4"
#!pip install "sentence-transformers == 2.5.1"
#!pip install "chromadb == 0.4.24"
#!pip install "wget == 3.2"

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget

filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# Use wget to download the file
wget.download(url, out=filename)
print('file downloaded')

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))

embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested')

model_id = 'google/flan-ul2'

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = "skills-network"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

flan_ul2_llm = WatsonxLLM(model=model)
