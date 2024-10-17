# Document Processing and QA Chatbot

This is a Streamlit-based chatbot application designed to process and answer questions from uploaded PDF documents. It uses language models from Hugging Face for question answering and FAISS for efficient similarity search.

## Features
- Upload and process PDF documents.
- Chunk documents for efficient processing.
- Create embeddings for document chunks.
- Answer questions based on the uploaded documents.

## Setup

### Prerequisites

- Python 3.7 or higher
- Pip

### Installation

1. **Clone the repository**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**

    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. **Start the Streamlit application**

    ```sh
    streamlit run app.py
    ```

2. **Open your browser**

    Streamlit will automatically open a new tab in your default web browser. If it doesn't, navigate to `http://localhost:8501` manually.

### Usage

1. **Upload Documents**

    - Go to the 'Upload Documents' section.
    - Upload one or multiple PDF documents.

2. **Chat with the Bot**

    - After the documents are processed, you can ask questions in the 'Chat with us' section.
    - The bot will respond with answers based on the uploaded documents.

## Testing

To test the application, you can use the following steps:

1. **Upload Test Documents**

    Use some sample PDF documents and upload them using the 'Upload Documents' section.

2. **Ask Questions**

    Ask questions related to the content of the uploaded documents in the chat section to verify the responses.

## File Structure

- `Content_Engine.py`: Main application file.
- `requirements.txt`: List of required Python packages.

## Dependencies

- `streamlit`: For building the web application.
- `torch`: For using PyTorch models.
- `transformers`: For loading and using Hugging Face models.
- `langchain`: For handling document processing and chunking.
- `faiss`: For efficient similarity search.
- `pandas`, `numpy`, `scikit-learn`: For data manipulation and machine learning tasks.

## Notes

- Ensure that the weights directory exists in the same directory as `Content_Engine.py` to store the downloaded models.
- Adjust the chunk size and overlap as needed based on the size and structure of your documents.

## Troubleshooting

- If the application fails to load models, ensure you have a stable internet connection.
- Check for any missing dependencies and install them using pip.

## Contributing

If you have suggestions for improvement or find bugs, feel free to open an issue or submit a pull request.
