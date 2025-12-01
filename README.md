# SecurityEngineeringResearchProject

Creating 2 LLM models and aligning each model to prevent the leakage of private training data. Will determine which alignment is more effective.

This repository contains a research-driven project completed as part of my Honors Colloquium in Security Engineering. The project focuses on model alignment, privacy preservation, and the detection of unintended information leakage in modern machine learning systems.

## Overview:
The central goal involves designing, training, and evaluating two separate LLM-based models. Each model is aligned through a different strategy to reduce the risk of private training data leakage during inference. After training, both models undergo adversarial evaluation to determine which alignment technique provides stronger privacy guarantees while maintaining model utility.

## Key Contributions
- Developed two custom LLM architectures based on one base CNN model using PyTorch and modern transformer frameworks.
- Implemented distinct alignment pipelines (including debiasing, instruction tuning, and safety-layer constraints).
- Designed tests to probe private-data leakage using simulated adversarial prompts.
- Conducted comparative analysis across accuracy, robustness, and privacy-preservation metrics.
- Documented strengths, weaknesses, and trade-offs between alignment strategies.

## Technologies & Tools
- Python, PyTorch, HuggingFace Transformers   
- Cryptographic randomness analysis  
- GPU-accelerated training and evaluation  
- Security principles related to confidentiality, model introspection, and alignment

## Purpose
This work bridges machine learning and security engineering, exploring how model design choices affect privacy risks. The goal is to better understand how to build aligned models that provide strong performance without compromising sensitive data.


