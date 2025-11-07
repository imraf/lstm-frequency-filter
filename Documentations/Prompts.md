# Prompts Used for PRD Generation

This document contains the prompts used to generate the LSTM Frequency Filter project.

---

## Prompt 1: Research PRD Structure and Best Practices

```
Research and provide comprehensive information about Product Requirements Document (PRD) structure and best practices. Include:

1. Standard sections that should be included in a professional PRD
2. Best practices for writing effective PRDs for machine learning/research projects
3. How to structure goals, success metrics, and requirements
4. Examples of good requirement specification formats (functional and non-functional)
5. How to handle risk assessment and mitigation in PRDs
6. Appropriate level of technical detail for ML projects
7. How to document assumptions, constraints, and scope boundaries
8. Best practices for documenting deliverables and timelines

Focus on PRDs for technical/ML research projects rather than traditional product development.
```

---

## Prompt 2: Generate PRD for LSTM Frequency Filter

```
Using the PRD best practices and structure you just provided, create a comprehensive Product Requirements Document (PRD) for the following project:

**Project**: LSTM Frequency Filter

**Context**: 
This is a machine learning research project that trains an LSTM neural network to extract specific frequency components from mixed signals. The project demonstrates applying deep learning to signal processing tasks.

**Key Information**:
- Uses PyTorch to implement a stacked LSTM architecture
- Generates synthetic signals with 4 frequencies (1, 3, 5, 7 Hz)
- Uses one-hot selectors to choose which frequency to extract
- Target performance: R² score ≥ 0.30
- 7 modular Python scripts from data generation through evaluation
- Fixed phase offsets for each frequency (0°, 45°, 90°, 135°)
- Adds Gaussian noise (σ=0.1) for realistic conditions
- 50-timestep sliding window sequences
- Model: 2-layer LSTM with 128 hidden units, dropout 0.2
- Training: Adam optimizer, batch size 64, MSE loss, ~50 epochs
- Comprehensive visualizations (10+) and complete pipeline automation

**Deliverables**:
- 7 Python scripts (generate_dataset, visualize_data, prepare_training_data, train_model, evaluate_model, summary, create_overview)
- Trained model achieving target performance
- Complete visualizations documenting all stages
- Full documentation

**Success Criteria**:
- R² ≥ 0.30 on test set
- Correlation ≥ 0.60
- Outperform baselines by >40%
- Complete reproducible pipeline

Create a thorough, professional PRD following industry best practices. Include all standard sections: executive summary, goals, user stories, functional requirements, technical requirements, non-functional requirements, risks, timeline, out-of-scope items, and appendices with mathematical formulations and technical details.
```

---

## Prompt 3: Implement Project According to PRD

```
Using the comprehensive PRD created above, implement the LSTM Frequency Filter project from scratch. Follow these guidelines:

**Implementation Approach**:
1. Create all 7 Python scripts as specified in the PRD deliverables
2. Implement the architecture exactly as documented (2-layer LSTM, 128 hidden units, dropout 0.2)
3. Follow the data generation specifications (4 frequencies at 1, 3, 5, 7 Hz with specified phase offsets)
4. Use the exact hyperparameters defined in the technical requirements
5. Implement all evaluation metrics and visualizations specified
6. Ensure complete end-to-end pipeline automation

**Key Requirements**:
- Follow all functional requirements from the PRD
- Meet all technical specifications (signal parameters, model architecture, training configuration)
- Implement all non-functional requirements (code quality, documentation, reproducibility)
- Create comprehensive visualizations at each stage
- Ensure the pipeline achieves the target performance metrics (R² ≥ 0.30)

**Code Quality Standards**:
- Write modular, well-structured code following Python best practices
- Include proper error handling and validation
- Use type hints where appropriate
- Keep functions focused and single-purpose
- Follow DRY principles

**Deliverables**:
- Complete implementation of all 7 scripts
- requirements.txt with all dependencies
- README.md with setup and usage instructions
- Shell script for running the complete pipeline
- All outputs (trained model, visualizations, metrics reports)

Start with script 01 (generate_dataset.py) and proceed sequentially through all scripts, ensuring each component works correctly before moving to the next. Test the complete pipeline end-to-end after implementation.
```

---

