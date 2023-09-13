# counselorGPT1.0
TASK: 
Build a chatbot with knowledge of the UCLA course registrar. Students should be able to query the chatbot about course offerings that match their interests, class prerequisites, and major/minor course requirements.
Implementation:
Step #1: From langchain.document_loaders, import UnstructuredPDFLoader. Download a PDF of the 2022-2023 UCLA course catalog (https://registrar.ucla.edu/file/b833e6ec-61b7-4b7e-961e-02af00520497).
Step #2: From langchain.text_splitter, import RecursiveCharacterTextSplitter. Split the UCLA catalog PDF into 400-character chunks with 200-character overlaps.
Step #2: Create embeddings of the ~39000 400-character chunks using OpenAI's Embeddings model. This will allow the application to match the student's query to relevant info in the catalog based on semantic similarity.
Step #4: Store embeddings vectors in a Pinecone cloud vector database.
Step #5: From langchain.llms, import OpenAI. Initialize LLM with custom OpenAI API Key.
Step #6: From langchain.chains.question_answering, import load_qa_chain. Perform a similairty search on the Pinecone vector database to retrieve the character chunks most relevant to the student's query. Then feed that chunk into a LangChain "chain" interface and call its "run" function.
Step #7: Get back a response capturing the essence of the relevant catalog chunk in the conversational style of a "GPT3.5-turbo"-powered chatbot.

Final product: https://counselorgpt.streamlit.app/
Application taken down in July 2023 after release of updated version.
