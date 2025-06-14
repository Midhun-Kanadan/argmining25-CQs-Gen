# ArgMining 2025 Submission – Team "gottfried-wilhelm-leibniz"

## Critical Question Generation (CQs-Gen)

This repository contains the complete pipeline developed by Team **gottfried-wilhelm-leibniz** for the **Critical Question Generation (CQs-Gen) Shared Task** at **ArgMining 2025**.

## Task Overview

The goal of the task is to generate **critical questions** that challenge the reasoning, assumptions, or conclusions in argumentative interventions. These questions are essential for fostering critical thinking and deepening understanding in argumentative discourse.

## Our Approach

We implemented a fully automated two-stage pipeline: first, prompting open-weight LLMs to generate candidate critical questions, and second, reranking them using a fine-tuned ModernBERT classifier to select the most useful ones. This combined approach consistently outperformed prompting-only baselines and achieved a test score of 0.569, ranking 5th overall in the ArgMining 2025 shared task.

## Repository Contents
```text
src/              # Core Python code for CQ generation and reranking
prompts/          # Prompt templates used for instructing LLMs
generated_cqs/    # Raw critical questions generated by each LLM
reranked_cqs/     # Top-3 useful CQs per intervention after reranking with ModernBERT
dataset/          # Sample, validation, and test sets provided by the shared task 
submission/       # Final prediction files submitted to the ArgMining 2025 shared task
README.md         # This readme file summarizing our submission and approach