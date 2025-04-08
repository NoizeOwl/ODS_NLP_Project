# Meeting Summarization using Large Language Models (LLMs)

This project explores the use of Large Language Models (LLMs) with up to 8 billion parameters for summarizing work meetings conducted in Russian. The main notebook, `main.ipynb`, contains the code for evaluating the performance of these models on two datasets: the publicly available rudialogsum v2 dataset and a custom synthetic dataset derived from real-world work calls.

## Project Overview

The project aims to develop a system for summarizing real automatic speech recognition (ASR) transcripts recorded during organizational meetings. It evaluates the performance of LLMs in terms of summarization accuracy using various metrics, including ROUGE scores.

## Key Features

1. Evaluation of multiple LLM models, including Cotype, T-lite-it-1.0-Q8-GGUF, and T-pro-it-1.0-Q4-K-M-GGUF.
2. Comparison of 0-shot and 1-shot prompt strategies for model inference.
3. Use of two datasets: rudialogsum-v2 and a custom synthetic dataset.
4. Implementation of various evaluation metrics: ROUGE, BLEU, METEOR, and BERTScore.

## Notebook Structure

The `main.ipynb` notebook is organized into several sections:

1. **Importing Libraries**: The notebook starts by importing necessary libraries and custom modules.

2. **Data Loading**: It loads the rudialogsum-v2 dataset and a custom synthetic dataset.

3. **Model Inference**: 
   - Implements inference for multiple LLM models (Cotype, T-lite, T-pro) using both 0-shot and 1-shot strategies.
   - Utilizes the `inference_model` module for model-specific inference functions.

4. **Metrics Calculation**: 
   - Calculates various evaluation metrics (ROUGE, BLEU, METEOR, BERTScore) using the `metrics` module.
   - Compares performance across different models and strategies.

5. **Synthetic Dataset Processing**: 
   - Includes code for processing a custom synthetic dataset.
   - Implements a modified 1-shot strategy for longer meeting transcripts.

6. **Results Analysis**: 
   - Saves predictions and metrics for further analysis.
   - Provides examples of generated summaries for comparison.

The notebook uses a modular approach, with custom functions in separate modules (`utils`, `metrics`, `inference_model`) to handle data processing, model inference, and metric calculation.


## Results

The project finds that the T-lite-it-1.0-Q8-GGUF model, operating in a one-shot mode, achieved the highest performance across both datasets, offering an optimal balance between summary quality and inference time.

## Future Work

1. Fine-tuning models for the specific task of meeting summarization.
2. Exploring multi-shot strategies for further improvement.
3. Developing a production-ready system for real-time meeting summarization.

## Acknowledgments

This project was conducted by Zakharov Artem, Abramenko Ilya, and Zhavoronkov Alexander as part of their research on LLM applications in natural language processing.

For more details and the full research paper, please refer to the file: https://github.com/NoizeOwl/ODS_NLP_Project/science_report.pdf
