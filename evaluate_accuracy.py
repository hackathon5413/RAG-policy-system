"""
Accuracy Testing and Evaluation Script for RAG Policy System
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Test questions for accuracy evaluation
TEST_QUESTIONS = [
    {
        "question": "What is the maximum coverage amount for hospitalization?",
        "category": "coverage_amounts",
        "expected_keywords": ["maximum", "coverage", "amount", "hospitalization", "limit"]
    },
    {
        "question": "Are pre-existing conditions covered?",
        "category": "exclusions",
        "expected_keywords": ["pre-existing", "conditions", "covered", "exclusion", "waiting"]
    },
    {
        "question": "What documents are required for claim submission?",
        "category": "claims_process",
        "expected_keywords": ["documents", "required", "claim", "submission", "forms"]
    },
    {
        "question": "Is maternity benefit included in the policy?",
        "category": "specific_benefits",
        "expected_keywords": ["maternity", "benefit", "included", "pregnancy", "delivery"]
    },
    {
        "question": "What is the waiting period for surgery coverage?",
        "category": "conditions",
        "expected_keywords": ["waiting", "period", "surgery", "coverage", "days", "months"]
    },
    {
        "question": "What are the exclusions for accidents?",
        "category": "exclusions",
        "expected_keywords": ["exclusions", "accidents", "not covered", "limitations"]
    },
    {
        "question": "How much is the premium for family coverage?",
        "category": "financial",
        "expected_keywords": ["premium", "family", "coverage", "amount", "cost"]
    },
    {
        "question": "What is the process for cashless treatment?",
        "category": "claims_process",
        "expected_keywords": ["cashless", "treatment", "process", "network", "hospital"]
    }
]

class AccuracyEvaluator:
    def __init__(self):
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
    async def evaluate_retrieval_accuracy(self, question: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evaluate retrieval accuracy for a single question"""
        
        try:
            from app.document_processor import enhanced_search_for_question
            from app.vector_store import get_embeddings
            
            start_time = time.time()
            
            # Get search results
            search_results = await enhanced_search_for_question(question)
            
            retrieval_time = time.time() - start_time
            
            # Analyze results
            total_results = len(search_results)
            
            if total_results == 0:
                return {
                    "question": question,
                    "total_results": 0,
                    "keyword_matches": 0,
                    "average_similarity": 0.0,
                    "retrieval_time": retrieval_time,
                    "accuracy_score": 0.0,
                    "issues": ["No results found"]
                }
            
            # Calculate keyword matching
            keyword_matches = 0
            similarities = []
            content_analysis = {
                "total_content_length": 0,
                "unique_sources": set(),
                "section_types": set()
            }
            
            for doc, score in search_results:
                content = doc.page_content.lower()
                metadata = doc.metadata
                
                # Count keyword matches
                for keyword in expected_keywords:
                    if keyword.lower() in content:
                        keyword_matches += 1
                
                # Collect similarity scores (convert distance to similarity)
                similarity = 1 - score if score <= 1 else 1 / (1 + score)
                similarities.append(similarity)
                
                # Content analysis
                content_analysis["total_content_length"] += len(content)
                content_analysis["unique_sources"].add(metadata.get("filename", "unknown"))
                content_analysis["section_types"].add(metadata.get("section_type", "unknown"))
            
            # Calculate metrics
            keyword_match_rate = keyword_matches / (len(expected_keywords) * total_results)
            average_similarity = sum(similarities) / len(similarities)
            
            # Calculate accuracy score (combination of keyword matching and similarity)
            accuracy_score = (keyword_match_rate * 0.4) + (average_similarity * 0.6)
            
            # Identify potential issues
            issues = []
            if keyword_match_rate < 0.3:
                issues.append("Low keyword match rate")
            if average_similarity < 0.7:
                issues.append("Low average similarity")
            if total_results < 5:
                issues.append("Few results returned")
            if len(content_analysis["unique_sources"]) < 2:
                issues.append("Results from limited sources")
            
            return {
                "question": question,
                "total_results": total_results,
                "keyword_matches": keyword_matches,
                "keyword_match_rate": keyword_match_rate,
                "average_similarity": average_similarity,
                "retrieval_time": retrieval_time,
                "accuracy_score": accuracy_score,
                "unique_sources": len(content_analysis["unique_sources"]),
                "section_types": list(content_analysis["section_types"]),
                "issues": issues,
                "top_3_similarities": similarities[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating question '{question}': {e}")
            return {
                "question": question,
                "error": str(e),
                "accuracy_score": 0.0
            }
    
    async def evaluate_answer_quality(self, question: str, answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evaluate answer quality"""
        
        answer_lower = answer.lower()
        
        # Check for keyword presence in answer
        keywords_in_answer = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        keyword_coverage = keywords_in_answer / len(expected_keywords)
        
        # Check answer structure and completeness
        structure_indicators = [
            "coverage", "exclusion", "condition", "requirement", 
            "amount", "limit", "process", "document"
        ]
        structure_score = sum(1 for indicator in structure_indicators if indicator in answer_lower) / len(structure_indicators)
        
        # Check for specific answer patterns
        has_direct_answer = any(pattern in answer_lower for pattern in [
            "yes", "no", "covered", "not covered", "included", "excluded", "amount", "limit"
        ])
        
        # Check for confidence indicators
        confidence_indicators = ["high", "medium", "low", "unclear", "insufficient information"]
        has_confidence = any(indicator in answer_lower for indicator in confidence_indicators)
        
        # Calculate quality score
        quality_factors = [
            keyword_coverage * 0.3,
            structure_score * 0.2,
            (1.0 if has_direct_answer else 0.0) * 0.3,
            (1.0 if has_confidence else 0.0) * 0.1,
            min(len(answer) / 500, 1.0) * 0.1  # Length appropriateness
        ]
        
        quality_score = sum(quality_factors)
        
        return {
            "question": question,
            "answer_length": len(answer),
            "keywords_in_answer": keywords_in_answer,
            "keyword_coverage": keyword_coverage,
            "structure_score": structure_score,
            "has_direct_answer": has_direct_answer,
            "has_confidence": has_confidence,
            "quality_score": quality_score
        }
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive accuracy evaluation"""
        
        print("🔍 Starting Comprehensive Accuracy Evaluation")
        print("=" * 60)
        
        retrieval_results = []
        answer_results = []
        overall_start = time.time()
        
        # Test each question
        for i, test_case in enumerate(TEST_QUESTIONS, 1):
            question = test_case["question"]
            category = test_case["category"]
            expected_keywords = test_case["expected_keywords"]
            
            print(f"\n📋 Test {i}/{len(TEST_QUESTIONS)}: {category}")
            print(f"❓ Question: {question}")
            
            # Test retrieval accuracy
            print("🔍 Testing retrieval...")
            retrieval_result = await self.evaluate_retrieval_accuracy(question, expected_keywords)
            retrieval_result["category"] = category
            retrieval_results.append(retrieval_result)
            
            # Test answer quality
            print("🤖 Testing answer generation...")
            try:
                from app.document_processor import answer_single_question
                answer = await answer_single_question(question)
                
                answer_result = await self.evaluate_answer_quality(question, answer, expected_keywords)
                answer_result["category"] = category
                answer_result["answer"] = answer[:200] + "..." if len(answer) > 200 else answer
                answer_results.append(answer_result)
                
                print(f"✅ Retrieval Score: {retrieval_result['accuracy_score']:.2f}")
                print(f"✅ Answer Quality: {answer_result['quality_score']:.2f}")
                
            except Exception as e:
                print(f"❌ Answer generation error: {e}")
                answer_results.append({
                    "question": question,
                    "category": category,
                    "error": str(e),
                    "quality_score": 0.0
                })
        
        total_time = time.time() - overall_start
        
        # Calculate overall metrics
        avg_retrieval_score = sum(r.get("accuracy_score", 0) for r in retrieval_results) / len(retrieval_results)
        avg_answer_quality = sum(r.get("quality_score", 0) for r in answer_results) / len(answer_results)
        avg_retrieval_time = sum(r.get("retrieval_time", 0) for r in retrieval_results) / len(retrieval_results)
        
        # Category-wise analysis
        category_scores = {}
        for category in set(t["category"] for t in TEST_QUESTIONS):
            cat_retrieval = [r for r in retrieval_results if r.get("category") == category]
            cat_answers = [r for r in answer_results if r.get("category") == category]
            
            category_scores[category] = {
                "avg_retrieval_score": sum(r.get("accuracy_score", 0) for r in cat_retrieval) / len(cat_retrieval),
                "avg_answer_quality": sum(r.get("quality_score", 0) for r in cat_answers) / len(cat_answers)
            }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(retrieval_results, answer_results)
        
        evaluation_report = {
            "timestamp": time.time(),
            "total_evaluation_time": total_time,
            "overall_metrics": {
                "average_retrieval_score": avg_retrieval_score,
                "average_answer_quality": avg_answer_quality,
                "average_retrieval_time": avg_retrieval_time,
                "total_questions_tested": len(TEST_QUESTIONS)
            },
            "category_analysis": category_scores,
            "detailed_results": {
                "retrieval_results": retrieval_results,
                "answer_results": answer_results
            },
            "recommendations": recommendations
        }
        
        # Save results
        results_file = f"./data/accuracy_evaluation_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"🎯 Overall Retrieval Accuracy: {avg_retrieval_score:.2f}/1.00")
        print(f"🎯 Overall Answer Quality: {avg_answer_quality:.2f}/1.00")
        print(f"⏱️  Average Retrieval Time: {avg_retrieval_time:.3f}s")
        print(f"📁 Results saved to: {results_file}")
        
        return evaluation_report
    
    def generate_recommendations(self, retrieval_results: List[Dict], answer_results: List[Dict]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Analyze retrieval issues
        low_accuracy_count = sum(1 for r in retrieval_results if r.get("accuracy_score", 0) < 0.6)
        if low_accuracy_count > len(retrieval_results) * 0.3:
            recommendations.append("Consider reducing chunk size for more focused retrieval")
            recommendations.append("Increase overlap between chunks for better context")
        
        # Analyze timing issues
        avg_retrieval_time = sum(r.get("retrieval_time", 0) for r in retrieval_results) / len(retrieval_results)
        if avg_retrieval_time > 2.0:
            recommendations.append("Consider caching embeddings to improve retrieval speed")
        
        # Analyze keyword matching
        avg_keyword_rate = sum(r.get("keyword_match_rate", 0) for r in retrieval_results) / len(retrieval_results)
        if avg_keyword_rate < 0.4:
            recommendations.append("Improve text preprocessing to better preserve key terms")
            recommendations.append("Consider using domain-specific embeddings")
        
        # Analyze answer quality
        avg_answer_quality = sum(r.get("quality_score", 0) for r in answer_results) / len(answer_results)
        if avg_answer_quality < 0.6:
            recommendations.append("Enhance prompt templates for more structured responses")
            recommendations.append("Provide more context in the retrieval phase")
        
        if not recommendations:
            recommendations.append("System performance is good! Consider fine-tuning for specific use cases.")
        
        return recommendations


async def main():
    """Main evaluation function"""
    evaluator = AccuracyEvaluator()
    
    try:
        await evaluator.run_comprehensive_evaluation()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
