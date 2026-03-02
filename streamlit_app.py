import streamlit as st
import os
from langchain_community.retrievers import OutlineRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Outline AI Chat", page_icon="📝")

# --- Load Secrets and Set Environment Variables ---
# Streamlit secrets are automatically available in st.secrets
# We map them to environment variables because LangChain components look for them there.
try:
    os.environ["OUTLINE_API_KEY"] = st.secrets["OUTLINE_API_KEY"]
    os.environ["OUTLINE_INSTANCE_URL"] = st.secrets["OUTLINE_INSTANCE_URL"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    # Validation flags
    has_secrets = True
except KeyError as e:
    st.error(f"Secret not found: {e}")
    st.info("Please set up .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    has_secrets = False

# --- Authentication ---
# Require a password before showing the chat
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Login Required")
    pwd = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        # compare against secret or hardcoded value
        expected = st.secrets.get("APP_PASSWORD", "")
        if pwd and expected and pwd == expected:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("パスワードが正しくありません。再試行してください。")
    st.stop()

# --- Sidebar ---
st.sidebar.title("Configuration")
st.sidebar.info("Using keys from Streamlit Secrets.")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.title("📝 Outline Wiki AI Chat")
st.markdown("Outline Wikiの記事を検索して、Geminiが回答します。")

# --- Initialize Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def get_rag_chain():
    """Build the RAG chain using the configured keys."""
    if not has_secrets:
        return None
        
    # Initialize Retriever (Uses OUTLINE_API_KEY and OUTLINE_INSTANCE_URL from env)
    retriever = OutlineRetriever(top_k=3)

    # Initialize Gemini (Uses GOOGLE_API_KEY from env)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # Prompt Template
    template = """
    あなたは社内ナレッジベースのアシスタントです。
    提供されたコンテキストのみを使用して質問に答えてください。
    答えがわからない場合は、「社内Wikiにはその情報が見当たりませんでした」と答えてください。
    日本語で、簡潔に分かりやすく回答してください。

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# --- Chat Interface ---
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt_input := st.chat_input("質問を入力してください..."):
    if not has_secrets:
        st.error("Secretsが設定されていないため実行できません。")
    else:
        # Add user message to UI and State
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("Outlineを検索して回答を生成中..."):
                try:
                    result = get_rag_chain()
                    if result:
                        chain, retriever = result
                        # Get documents directly from retriever
                        retrieved_docs = retriever.invoke(prompt_input)
                        st.session_state.retrieved_docs = retrieved_docs
                        
                        # Get response from chain
                        response = chain.invoke(prompt_input)
                        st.markdown(response)
                        
                        # Display referenced articles
                        if "retrieved_docs" in st.session_state and st.session_state.retrieved_docs:
                            st.markdown("---")
                            st.markdown("**参考資料:**")
                            for i, doc in enumerate(st.session_state.retrieved_docs, 1):
                                # Debug: show metadata structure
                                metadata = getattr(doc, 'metadata', {})
                                
                                # Try different metadata keys for URL
                                url = None
                                title = f"記事 {i}"
                                
                                if isinstance(metadata, dict):
                                    # Try common URL keys
                                    url = metadata.get('url') or metadata.get('source') or metadata.get('link') or metadata.get('href')
                                    # Try title keys
                                    title = metadata.get('title') or metadata.get('page_title') or metadata.get('name') or title
                                
                                if url:
                                    st.markdown(f"[{title}]({url})")
                                else:
                                    st.markdown(f"• {title}")
                        
                        # Add assistant response to State
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"エラーが発生しました: {str(e)}"
                    st.error(error_msg)