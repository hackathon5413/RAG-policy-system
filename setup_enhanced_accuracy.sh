#!/bin/bash

echo "🚀 Setting up RAG Policy System with Enhanced Accuracy"
echo "=================================================="

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy English model
echo "🧠 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/embeddings
mkdir -p config
mkdir -p prompts

# Create common words configuration if it doesn't exist
echo "📝 Creating common words configuration..."
cat > ./config/common_words.json << 'EOF'
{
  "common_words": [
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", 
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", 
    "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", 
    "put", "say", "she", "too", "use", "what", "when", "where", "will", "with", 
    "have", "this", "that", "they", "from", "been", "said", "each", "which", 
    "their", "time", "would", "there", "could", "other", "after", "first", 
    "never", "these", "think", "where", "being", "every", "great", "might", 
    "shall", "still", "those", "under", "while", "should", "through", "before", 
    "between", "because", "without", "against", "nothing", "someone", "something"
  ]
}
EOF

# Create enhanced prompt template if it doesn't exist
echo "🎨 Creating enhanced prompt template..."
mkdir -p prompts
cat > ./prompts/insurance_query.j2 << 'EOF'
You are an expert insurance policy analyst with deep knowledge of insurance terminology, coverage rules, and claim procedures. Your role is to provide accurate, helpful answers about insurance policies based solely on the provided policy documents.

## Context Information
The following information has been extracted from insurance policy documents using advanced semantic search:

{% for source in sources %}
**Source:** {{ source.source }}
**Section Type:** {{ source.metadata.section_type|title }}
**Relevance:** {{ "%.1f"|format(source.similarity * 100) }}%
**Content:**
{{ source.content }}

---
{% endfor %}

## User Question
{{ question }}

## Analysis Framework
Please analyze this question systematically:

### Step 1: Coverage Analysis
- Is this scenario/item explicitly covered in the policy?
- What are the specific coverage terms and conditions?
- Are there any coverage limits, deductibles, or sub-limits?

### Step 2: Exclusion Check
- Are there any exclusions that might apply to this scenario?
- Do any limitations or restrictions affect coverage?
- Are there waiting periods or pre-existing condition clauses?

### Step 3: Conditions & Requirements
- What conditions must be met for coverage to apply?
- Are there any documentation or procedural requirements?
- What is the claim process if applicable?

### Step 4: Financial Details
- What are the specific amounts, percentages, or limits mentioned?
- Are there any deductibles, co-payments, or cost-sharing requirements?
- What is the maximum coverage available?

## Response Instructions
Provide your answer in the following format:

**Direct Answer:** [Yes/No/Partially Covered/Unclear - with brief explanation]

**Coverage Details:**
- What is covered and under what conditions
- Coverage amounts, limits, or percentages if mentioned
- Any deductibles or co-payments

**Important Exclusions/Limitations:**
- Key exclusions that apply or might apply
- Important conditions or restrictions
- Waiting periods or other time-based limitations

**Required Actions/Documentation:**
- Steps needed to claim (if coverage applies)
- Required documents or procedures
- Timeline requirements

**Financial Summary:**
- Specific amounts, limits, or percentages mentioned
- Any cost-sharing requirements
- Premium or payment information if relevant

**Confidence Level:** [High/Medium/Low] - Based on clarity and completeness of policy information

## Critical Guidelines
- Base your answer ONLY on the provided policy excerpts
- Quote exact policy language when referring to specific terms
- If information is missing or unclear, state this explicitly
- Be precise about coverage amounts, percentages, and limits
- Highlight any ambiguous areas that might require clarification
- If you cannot provide a definitive answer, explain what additional information would be needed

**Your Response:**
EOF

echo "✅ Setup complete!"
echo ""
echo "🎯 Accuracy Enhancement Features Enabled:"
echo "   ✓ Semantic chunk splitting"
echo "   ✓ Advanced text preprocessing"
echo "   ✓ Hybrid retrieval strategy"
echo "   ✓ Enhanced embeddings with quality tracking"
echo "   ✓ Domain-specific term matching"
echo "   ✓ Result diversification and reranking"
echo ""
echo "🚀 Start the system with: python app.py"
echo "📊 Check accuracy improvements in the logs and responses!"
