# Decoder-Only Transformer: A Simple LLM Implementation  

This repository contains an implementation of a **Decoder-Only Transformer**, a foundational architecture for large language models (LLMs). The model processes input text by passing it through several key components:  

- **Word Embedding:** Converts input tokens (e.g., words) into dense vector representations.  
- **Positional Encoding:** Adds positional information to embeddings, allowing the model to understand token order.  
- **Masked Self-Attention:** Focuses on previous tokens during training to enable autoregressive text generation.  
- **Residual Connections and Fully Connected Layers:** Enhance model efficiency and depth for better learning.  
- **Softmax Layer:** Outputs probabilities for the next token in the sequence.  

This implementation is ideal for experimenting with LLM concepts and serves as a learning resource for understanding decoder-only transformer models.  
I have used PyTorch + Lightning to create and optimize a Decoder-Only Transformer.

Feel free to explore, modify, and contribute!  
