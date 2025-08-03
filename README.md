# BaZi-GPT Fine-tuning Project

A specialized fine-tuned GPT model for Four Pillars of Destiny (八字) consultation and analysis.

## 📋 Overview

This project creates a fine-tuned GPT-3.5-turbo model specialized in Chinese BaZi (Four Pillars of Destiny) fortune-telling and numerology. The model provides bilingual responses (Chinese + English) with structured analysis and cultural accuracy.

## 🎯 Features

- **Bilingual Responses**: Provides answers in both Chinese and English
- **Structured Analysis**: Organized output with clear sections
- **Cultural Accuracy**: Preserves traditional Chinese metaphysical concepts
- **Professional Format**: Includes appropriate disclaimers and formatting
- **Comprehensive Coverage**: Handles theory, chart analysis, and practical guidance

## 📁 Project Structure

```
bazi-gpt-finetune/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── config/
│   ├── system_prompt.txt       # System prompt for the model
│   └── hyperparameters.json    # Training hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_processor.py       # Data processing utilities
│   ├── fine_tuning.py          # Fine-tuning workflow
│   ├── model_tester.py         # Model testing and evaluation
│   └── utils.py                # Helper functions
├── data/
│   └── sample_data.csv         # Sample training data format
├── tests/
│   └── test_questions.json     # Test questions for evaluation
└── docs/
    ├── SETUP.md                # Detailed setup instructions
    ├── API_USAGE.md            # API usage examples
    └── FINE_TUNING_GUIDE.md    # Fine-tuning guide
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/skkuhg/bazi-gpt-finetune.git
cd bazi-gpt-finetune

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Prepare Your Data

Place your BaZi Q&A dataset in CSV format with columns: `question`, `answer`

### 3. Fine-tune the Model

```python
from src.fine_tuning import BaZiFineTuner

# Initialize fine-tuner
tuner = BaZiFineTuner(api_key="your-openai-api-key")

# Process data and create fine-tuning job
tuner.prepare_data("data/your_dataset.csv")
job_id = tuner.start_fine_tuning()

# Monitor training
model_id = tuner.monitor_job(job_id)
```

### 4. Test Your Model

```python
from src.model_tester import BaZiModelTester

# Test the fine-tuned model
tester = BaZiModelTester(model_id="your-fine-tuned-model-id")
results = tester.run_comprehensive_test()
```

## 📊 Model Performance

Our fine-tuned model achieves:
- **Success Rate**: 100% response generation
- **Quality Score**: 0.73/1.0 average across test categories
- **Bilingual Capability**: Native Chinese + English responses
- **Structured Output**: Professional formatting with clear sections

## 🔧 Configuration

### System Prompt
The model uses a specialized system prompt that ensures:
- Bilingual responses (Chinese first, English after)
- Structured output with numbered sections
- Cultural accuracy and appropriate disclaimers
- Professional tone without fatalistic language

### Hyperparameters
- Base Model: `gpt-3.5-turbo-0125`
- Epochs: 3
- Batch Size: 1
- Learning Rate Multiplier: 0.3

## 📚 Documentation

- [Setup Guide](docs/SETUP.md) - Detailed installation and setup
- [API Usage](docs/API_USAGE.md) - How to use the fine-tuned model
- [Fine-tuning Guide](docs/FINE_TUNING_GUIDE.md) - Complete fine-tuning workflow

## 🧪 Testing

The project includes comprehensive testing:
- **Basic Theory**: Five Elements, Ten Gods, etc.
- **Chart Analysis**: Birth chart interpretation
- **Historical Context**: Traditional numerology background
- **Practical Application**: Career, relationship guidance

## 🔒 Privacy & Security

- API keys are stored in environment variables
- No sensitive data in repository
- Secure handling of personal birth information

## 📈 Usage Examples

### Basic Query
```python
response = client.chat.completions.create(
    model="your-fine-tuned-model-id",
    messages=[
        {"role": "user", "content": "What are the Five Elements in BaZi?"}
    ]
)
```

### Chart Analysis
```python
response = client.chat.completions.create(
    model="your-fine-tuned-model-id", 
    messages=[
        {"role": "user", "content": "1990-05-03 09:28 GMT+8 Please analyze my BaZi chart"}
    ]
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This model is for educational and entertainment purposes. BaZi readings should not be used as the sole basis for important life decisions. Always consult qualified professionals for medical, legal, or financial advice.

## 🙏 Acknowledgments

- Traditional Chinese metaphysics and BaZi practitioners
- OpenAI for the fine-tuning capabilities
- The open-source community for tools and libraries

---

**Note**: This is a specialized AI model for Chinese traditional practices. Please use responsibly and with cultural sensitivity.