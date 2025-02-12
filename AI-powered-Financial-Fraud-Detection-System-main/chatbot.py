import streamlit as st
import random

class AtlasBot:
    def __init__(self):
        self.responses = {
            "hello": [
                "Hello! I'm ATLAS Assistant. How can I help you today?",
                "Hi there! Ready to help you with fraud detection.",
                "Welcome! What would you like to know about ATLAS?"
            ],
            "help": [
                "I can help you with navigation, fraud detection, or transaction analysis. What interests you?",
                "Need assistance? I can explain features, guide you through analysis, or help with navigation.",
                "How can I assist? Ask me about features, analysis, or using ATLAS."
            ],
            "navigation": [
                "Click 'Try Now' to test our fraud detection system.",
                "Want to analyze a transaction? Use the 'Try Now' button above.",
                "Head to the analysis page using the 'Try Now' button to test transactions."
            ],
            "features": [
                "ATLAS offers real-time fraud detection with 99.9% accuracy and risk assessment.",
                "Our system provides instant fraud detection, risk analysis, and forensic insights.",
                "Key features include real-time analysis, <0.1% false positives, and detailed risk assessment."
            ],
            "analysis": [
                "Enter transaction details in the analysis page to check for fraud risks.",
                "To analyze a transaction, provide the account details and amount in the analysis section.",
                "Use the analysis page to input transaction information and get instant risk assessment."
            ],
            "default": [
                "Try asking about features, navigation, or fraud detection.",
                "I can help with ATLAS features, usage, or fraud detection. What would you like to know?",
                "Not sure about that. Ask me about features, analysis, or how to use ATLAS."
            ]
        }
        if "response_history" not in st.session_state:
            st.session_state.response_history = {}

    def get_response(self, query: str) -> str:
        query = query.lower()
        
        # Determine response category
        if "hello" in query or "hi" in query:
            category = "hello"
        elif "help" in query:
            category = "help"
        elif "navigate" in query or "use" in query:
            category = "navigation"
        elif "feature" in query or "what" in query:
            category = "features"
        elif "analysis" in query or "detect" in query:
            category = "analysis"
        else:
            category = "default"
        
        # Initialize category history if not exists
        if category not in st.session_state.response_history:
            st.session_state.response_history[category] = []
        
        # Get unused responses
        used_responses = st.session_state.response_history[category]
        available_responses = [r for r in self.responses[category] if r not in used_responses]
        
        # If all responses used, reset history
        if not available_responses:
            st.session_state.response_history[category] = []
            available_responses = self.responses[category]
        
        # Select random response
        response = random.choice(available_responses)
        
        # Update history
        st.session_state.response_history[category].append(response)
        
        return response

def render_chatbot():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "bot" not in st.session_state:
        st.session_state.bot = AtlasBot()

    # Chat styling
    st.markdown("""
        <style>
        .chat-container {
            background-color: var(--accent);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .user-message {
            background-color: rgba(0, 220, 130, 0.1);
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            text-align: right;
        }
        .bot-message {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

    # Chat input with a submit button
    with st.form(key="chat_form"):
        user_input = st.text_input("Ask me anything about ATLAS...", key="chat_input")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            response = st.session_state.bot.get_response(user_input)
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat
            st.rerun()