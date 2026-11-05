# MindChat – Local AI Chatbot with LangChain Memory

## Overview

MindChat is a locally running AI chatbot developed using Python, Streamlit, Ollama, and LangChain. It provides an interactive conversational interface that can generate intelligent responses while maintaining conversation context through LangChain memory.

The application uses a locally hosted Large Language Model (LLM), allowing users to interact with an AI assistant without depending on external AI APIs.

## Project Objectives

- Develop a simple and interactive AI chatbot.
- Integrate a locally running Large Language Model.
- Implement conversational memory using LangChain.
- Maintain context across multiple messages.
- Provide a professional and user-friendly interface.
- Enable users to manage and export conversations.

## Features

- Local AI inference using Ollama.
- Llama 3.2 language model integration.
- Conversation memory using LangChain.
- Context-aware responses.
- Real-time response streaming.
- New conversation functionality.
- Clear current conversation functionality.
- Download conversation history.
- Response generation time display.
- Professional dark-themed user interface.
- Local processing without external AI API keys.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Streamlit | Web-based user interface |
| Ollama | Local LLM execution |
| Llama 3.2 | Language model |
| LangChain | LLM integration and prompt management |
| LangChain Memory | Conversation history management |

## Project Structure

```text
MindChat/
│
├── app.py
├── requirements.txt
└── README.md
```
## LangChain Memory Implementation

MindChat uses LangChain's `RunnableWithMessageHistory` and `InMemoryChatMessageHistory` to maintain conversational context.

### Memory Workflow

1. The user sends a message.
2. The message is passed to the LangChain chat chain.
3. Previous conversation messages are retrieved.
4. The conversation history is included in the prompt.
5. Ollama generates a response.
6. The user message and assistant response are stored in memory.
7. The next message uses the previous conversation context.

Memory is maintained separately for each conversation session.

The application uses in-memory storage, so conversation history is temporary and is not permanently stored after the application session ends.

## Testing

The following test cases can be used to verify the application.

| Test Case | Description | Expected Result |
|---|---|---|
| Basic Chat | Send a simple greeting | Chatbot generates a response |
| Conversation Memory | Provide a name and ask it later | Chatbot remembers the name |
| Context Awareness | Ask follow-up questions | Chatbot understands previous context |
| Coding Assistance | Request a Python program | Chatbot generates code |
| New Conversation | Start a new session | Previous conversation memory is not used |
| Clear Chat | Clear the current conversation | Current history is removed |
| Local Model | Run chatbot using Ollama | Responses generated locally |
| Export Conversation | Download chat history | Text file is generated |

## Privacy and Security

- AI inference is performed locally using Ollama.
- No external AI API key is required.
- Conversation history is maintained in application memory.
- Chat history is not automatically persisted to a database.
- Local inference reduces dependence on external AI services.

## Limitations

- Response generation depends on available system hardware.
- Large language models may require significant RAM.
- Conversation memory is temporary.
- The chatbot may occasionally generate inaccurate information.
- Model performance depends on the installed LLM.

## Future Enhancements

- Persistent conversation storage using SQLite.
- Multiple local LLM support.
- User authentication.
- Conversation history management.
- Document-based question answering.
- Voice input and output.
- Improved chatbot personalization.

## Conclusion

MindChat demonstrates the integration of local Large Language Models with LangChain conversational memory in an interactive chatbot application.

The project combines local AI inference, context-aware conversations, and a modern interface to provide an accessible and privacy-conscious conversational AI system.
