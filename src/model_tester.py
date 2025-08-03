"""
Model testing and evaluation for fine-tuned BaZi-GPT.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from .utils import load_env_vars, validate_api_key, setup_logging


class BaZiModelTester:
    """Test and evaluate fine-tuned BaZi-GPT models."""
    
    def __init__(self, model_id: str, api_key: str = None):
        """
        Initialize the model tester.
        
        Args:
            model_id: Fine-tuned model ID to test
            api_key: OpenAI API key (if not provided, will load from environment)
        """
        self.logger = setup_logging()
        
        if not api_key:
            env_vars = load_env_vars()
            api_key = env_vars['api_key']
        
        if not validate_api_key(api_key):
            raise ValueError("Invalid or missing OpenAI API key")
        
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.test_results = []
        
        self.logger.info(f"Model tester initialized for model: {model_id}")
    
    def load_test_questions(self, test_file: str) -> List[Dict[str, Any]]:
        """
        Load test questions from JSON file.
        
        Args:
            test_file: Path to test questions file
            
        Returns:
            List of test questions
        """
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            self.logger.info(f"Loaded {len(questions)} test questions from {test_file}")
            return questions
        except FileNotFoundError:
            self.logger.warning(f"Test file not found: {test_file}, using default questions")
            return self._get_default_test_questions()
    
    def _get_default_test_questions(self) -> List[Dict[str, Any]]:
        """Get default test questions if file not found."""
        return [
            {
                "category": "Basic Theory",
                "question": "什麼是八字中的五行？請解釋金木水火土的特性。",
                "expected_elements": ["五行", "金木水火土", "特性", "bilingual"]
            },
            {
                "category": "Chart Analysis", 
                "question": "1990-05-03 09:28 GMT+8 請分析我的八字命盤",
                "expected_elements": ["八字", "命盤", "分析", "1990", "bilingual"]
            },
            {
                "category": "Ten Gods",
                "question": "十神中的正財和偏財有什麼區別？",
                "expected_elements": ["十神", "正財", "偏財", "區別", "bilingual"]
            },
            {
                "category": "Career Guidance",
                "question": "八字中如何看適合的職業方向？",
                "expected_elements": ["職業", "方向", "八字", "建議", "bilingual"]
            },
            {
                "category": "Relationship",
                "question": "八字合婚需要看哪些方面？",
                "expected_elements": ["合婚", "方面", "配對", "關係", "bilingual"]
            }
        ]
    
    def test_single_question(self, question: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Test the model with a single question.
        
        Args:
            question: Question to test
            max_tokens: Maximum tokens for response
            
        Returns:
            Test result dictionary
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": question}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                "question": question,
                "response": response.choices[0].message.content,
                "response_time": response_time,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "success": True,
                "error": None
            }
            
            self.logger.info(f"Question tested successfully. Response time: {response_time:.2f}s")
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                "question": question,
                "response": None,
                "response_time": response_time,
                "tokens_used": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "success": False,
                "error": str(e)
            }
            
            self.logger.error(f"Error testing question: {e}")
            return result
    
    def evaluate_response_quality(self, result: Dict[str, Any], 
                                expected_elements: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a response.
        
        Args:
            result: Test result from test_single_question
            expected_elements: List of expected elements in response
            
        Returns:
            Quality evaluation metrics
        """
        if not result["success"] or not result["response"]:
            return {
                "overall_score": 0.0,
                "completeness": 0.0,
                "structure": 0.0,
                "bilingual": 0.0,
                "relevance": 0.0,
                "errors": ["No response generated"]
            }
        
        response = result["response"]
        errors = []
        
        # Check for bilingual content (Chinese + English)
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
        has_english = any(char.isalpha() and ord(char) < 128 for char in response)
        bilingual_score = 1.0 if (has_chinese and has_english) else 0.5 if has_chinese else 0.0
        
        if not has_chinese:
            errors.append("Missing Chinese content")
        if not has_english:
            errors.append("Missing English translation")
        
        # Check for expected elements
        completeness_score = 0.0
        if expected_elements:
            found_elements = sum(1 for element in expected_elements 
                               if element.lower() in response.lower())
            completeness_score = found_elements / len(expected_elements)
        else:
            completeness_score = 0.8  # Default score if no expected elements
        
        # Check response structure (numbered sections, clear organization)
        structure_indicators = [
            any(char.isdigit() and '.' in response for char in response[:100]),  # Numbered points
            len(response.split('\n')) > 3,  # Multiple paragraphs
            '：' in response or ':' in response,  # Section separators
            len(response) > 100  # Sufficient length
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)
        
        # Check relevance (basic keyword matching)
        bazi_keywords = ['八字', '命理', '五行', '十神', 'BaZi', 'Four Pillars']
        has_bazi_content = any(keyword in response for keyword in bazi_keywords)
        relevance_score = 1.0 if has_bazi_content else 0.3
        
        if not has_bazi_content:
            errors.append("Missing BaZi-related content")
        
        # Calculate overall score
        overall_score = (
            bilingual_score * 0.3 +
            completeness_score * 0.3 +
            structure_score * 0.2 +
            relevance_score * 0.2
        )
        
        evaluation = {
            "overall_score": overall_score,
            "completeness": completeness_score,
            "structure": structure_score,
            "bilingual": bilingual_score,
            "relevance": relevance_score,
            "errors": errors,
            "response_length": len(response),
            "has_chinese": has_chinese,
            "has_english": has_english
        }
        
        return evaluation
    
    def run_test_suite(self, test_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the complete test suite.
        
        Args:
            test_questions: List of test questions to run
            
        Returns:
            List of test results with evaluations
        """
        self.logger.info(f"Running test suite with {len(test_questions)} questions")
        results = []
        
        for i, test_case in enumerate(test_questions, 1):
            self.logger.info(f"Testing question {i}/{len(test_questions)}: {test_case['category']}")
            
            # Test the question
            result = self.test_single_question(test_case["question"])
            
            # Evaluate the response
            evaluation = self.evaluate_response_quality(
                result, 
                test_case.get("expected_elements", [])
            )
            
            # Combine results
            test_result = {
                **result,
                "category": test_case["category"],
                "evaluation": evaluation
            }
            
            results.append(test_result)
            
            # Brief pause between requests
            time.sleep(1)
        
        self.test_results = results
        self.logger.info("Test suite completed")
        return results
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive test report.
        
        Args:
            results: Test results from run_test_suite
            
        Returns:
            Test report with statistics and analysis
        """
        if not results:
            return {"error": "No test results to analyze"}
        
        # Calculate overall statistics
        successful_tests = [r for r in results if r["success"]]
        success_rate = len(successful_tests) / len(results)
        
        if successful_tests:
            avg_response_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
            avg_tokens = sum(r["tokens_used"] for r in successful_tests) / len(successful_tests)
            
            # Quality metrics
            evaluations = [r["evaluation"] for r in successful_tests]
            avg_overall_score = sum(e["overall_score"] for e in evaluations) / len(evaluations)
            avg_completeness = sum(e["completeness"] for e in evaluations) / len(evaluations)
            avg_structure = sum(e["structure"] for e in evaluations) / len(evaluations)
            avg_bilingual = sum(e["bilingual"] for e in evaluations) / len(evaluations)
            avg_relevance = sum(e["relevance"] for e in evaluations) / len(evaluations)
        else:
            avg_response_time = 0
            avg_tokens = 0
            avg_overall_score = 0
            avg_completeness = 0
            avg_structure = 0
            avg_bilingual = 0
            avg_relevance = 0
        
        # Category breakdown
        categories = {}
        for result in results:
            category = result["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        category_stats = {}
        for category, cat_results in categories.items():
            cat_successful = [r for r in cat_results if r["success"]]
            if cat_successful:
                cat_avg_score = sum(r["evaluation"]["overall_score"] for r in cat_successful) / len(cat_successful)
            else:
                cat_avg_score = 0
            
            category_stats[category] = {
                "total_tests": len(cat_results),
                "successful": len(cat_successful),
                "success_rate": len(cat_successful) / len(cat_results),
                "avg_score": cat_avg_score
            }
        
        # Common errors
        all_errors = []
        for result in results:
            if result["success"]:
                all_errors.extend(result["evaluation"]["errors"])
        
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        report = {
            "model_id": self.model_id,
            "test_timestamp": time.time(),
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "avg_tokens_used": avg_tokens
            },
            "quality_metrics": {
                "overall_score": avg_overall_score,
                "completeness": avg_completeness,
                "structure": avg_structure,
                "bilingual_capability": avg_bilingual,
                "relevance": avg_relevance
            },
            "category_breakdown": category_stats,
            "common_errors": error_counts,
            "detailed_results": results
        }
        
        # Log summary
        self.logger.info("=== Test Report Summary ===")
        self.logger.info(f"Success Rate: {success_rate:.1%}")
        self.logger.info(f"Overall Quality Score: {avg_overall_score:.2f}")
        self.logger.info(f"Average Response Time: {avg_response_time:.2f}s")
        self.logger.info(f"Bilingual Capability: {avg_bilingual:.2f}")
        
        return report
    
    def save_test_results(self, results: List[Dict[str, Any]], 
                         output_file: str = "output/test_results.json") -> None:
        """
        Save test results to file.
        
        Args:
            results: Test results to save
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_test_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Test results saved to {output_file}")
    
    def run_comprehensive_test(self, test_file: str = "tests/test_questions.json",
                             output_file: str = "output/test_results.json") -> Dict[str, Any]:
        """
        Run a comprehensive test of the model.
        
        Args:
            test_file: Path to test questions file
            output_file: Path to save results
            
        Returns:
            Complete test report
        """
        self.logger.info("Starting comprehensive model test")
        
        # Load test questions
        test_questions = self.load_test_questions(test_file)
        
        # Run tests
        results = self.run_test_suite(test_questions)
        
        # Generate and save report
        self.save_test_results(results, output_file)
        
        # Return the report
        report = self.generate_test_report(results)
        
        self.logger.info("Comprehensive test completed")
        return report