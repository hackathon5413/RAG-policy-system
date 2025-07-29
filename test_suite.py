#!/usr/bin/env python3
"""
Detailed Test Suite for Policy RAG System
Runs all 6 test cases sequentially with comprehensive request/response logging
"""

import asyncio
import json
import time
import logging
from datetime import datetime
import sys
import os
import argparse
from typing import Dict, Any, List

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from app.document_processor import process_document_and_answer

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestSuite')

# Suppress verbose logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('app.embeddings').setLevel(logging.WARNING)

class TestRunner:
    def __init__(self):
        self.test_cases = [
            {
                "name": "Test Case 1 - Arogya Sanjeevani Policy",
                "document_url": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
                "questions": [
                    "When will my root canal claim of Rs 25,000 be settled?",
                    "I have done an IVF for Rs 56,000. Is it covered?",
                    "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?",
                    "Give me a list of documents to be uploaded for hospitalization for heart surgery."
                ]
            },
            {
                "name": "Test Case 2 - Arogya Sanjeevani Claim Balance",
                "document_url": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
                "questions": [
                    "I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?"
                ]
            },
            {
                "name": "Test Case 3 - Super Splendor Manual",
                "document_url": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
                "questions": [
                    "What is the ideal spark plug gap recommeded",
                    "Does this comes in tubeless tyre version",
                    "Is it compulsoury to have a disc brake",
                    "Can I put thums up instead of oil",
                    "Give me JS code to generate a random number between 1 and 100"
                ]
            },
            {
                "name": "Test Case 4 - Family Medicare Policy",
                "document_url": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
                "questions": [
                    "Is Non-infective Arthritis covered?",
                    "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?",
                    "Is abortion covered?"
                ]
            },
            {
                "name": "Test Case 5 - Indian Constitution",
                "document_url": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
                "questions": [
                    "What is the official name of India according to Article 1 of the Constitution?",
                    "Which Article guarantees equality before the law and equal protection of laws to all persons?",
                    "What is abolished by Article 17 of the Constitution?",
                    "What are the key ideals mentioned in the Preamble of the Constitution of India?",
                    "Under which Article can Parliament alter the boundaries, area, or name of an existing State?",
                    "According to Article 24, children below what age are prohibited from working in hazardous industries like factories or mines?",
                    "What is the significance of Article 21 in the Indian Constitution?",
                    "Article 15 prohibits discrimination on certain grounds. However, which groups can the State make special provisions for under this Article?",
                    "Which Article allows Parliament to regulate the right of citizenship and override previous articles on citizenship (Articles 5 to 10)?",
                    "What restrictions can the State impose on the right to freedom of speech under Article 19(2)?"
                ]
            },
            {
                "name": "Test Case 6 - Indian Constitution (Legal Scenarios)",
                "document_url": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
                "questions": [
                    "If my car is stolen, what case will it be in law?",
                    "If I am arrested without a warrant, is that legal?",
                    "If someone denies me a job because of my caste, is that allowed?",
                    "If the government takes my land for a project, can I stop it?",
                    "If my child is forced to work in a factory, is that legal?",
                    "If I am stopped from speaking at a protest, is that against my rights?",
                    "If a religious place stops me from entering because I'm a woman, is that constitutional?",
                    "If I change my religion, can the government stop me?",
                    "If the police torture someone in custody, what right is being violated?",
                    "If I'm denied admission to a public university because I'm from a backward community, can I do something?"
                ]
            },
            {
                "name": "Test Case 7 - Newton's Principia",
                "document_url": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
                "questions": [
                    "How does Newton define 'quantity of motion' and how is it distinct from 'force'?",
                    "According to Newton, what are the three laws of motion and how do they apply in celestial mechanics?",
                    "How does Newton derive Kepler's Second Law (equal areas in equal times) from his laws of motion and gravitation?",
                    "How does Newton demonstrate that gravity is inversely proportional to the square of the distance between two masses?",
                    "What is Newton's argument for why gravitational force must act on all masses universally?",
                    "How does Newton explain the perturbation of planetary orbits due to other planets?",
                    "What mathematical tools did Newton use in Principia that were precursors to calculus, and why didn't he use standard calculus notation?",
                    "How does Newton use the concept of centripetal force to explain orbital motion?",
                    "How does Newton handle motion in resisting media, such as air or fluids?",
                    "In what way does Newton's notion of absolute space and time differ from relative motion, and how does it support his laws?",
                    "Who was the grandfather of Isaac Newton?",
                    "Do we know any other descent of Isaac Newton apart from his grandfather?"
                ]
            }
        ]

    def print_separator(self, char="=", length=80):
        print(char * length)

    def print_test_header(self, test_name, test_num, total_tests):
        self.print_separator()
        print(f"🧪 Running {test_name} ({test_num}/{total_tests})")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Starting test {test_num}/{total_tests}: {test_name}")
        self.print_separator("-")

    def print_detailed_request(self, test_case: Dict[str, Any], test_num: int):
        """Print request information"""
        print(f"📋 REQUEST (Test {test_num}): {len(test_case['questions'])} questions")
        print(f"📄 Document: {test_case['document_url'][:60]}...\n")
        
        # Show all questions
        for i, question in enumerate(test_case['questions'], 1):
            print(f"   {i}. {question}")
        
        # Log to file (keep detailed logging in file)
        logger.info(f"REQUEST - Test {test_num}")
        logger.info(f"Document URL: {test_case['document_url']}")
        logger.info(f"Questions ({len(test_case['questions'])}): {json.dumps(test_case['questions'], indent=2)}")

    def print_detailed_response(self, result: Dict[str, Any], duration: float, test_num: int):
        """Print response information with full answers"""
        print(f"\n📊 RESPONSE DETAILS (Test {test_num}):")
        print(f"⏱️  Duration: {duration:.2f}s | ✅ Success: {result['success']}")
        
        if result['success']:
            answers = result['answers']
            print(f"📝 Generated {len(answers)} answers:\n")
            
            for i, answer in enumerate(answers, 1):
                if answer and answer.strip():
                    print(f"   {i}. {answer}")
                else:
                    print(f"   {i}. ❌ EMPTY")
                print()  # Add spacing between answers
            
            # Print document info if available
            if 'document_info' in result:
                doc_info = result['document_info']
                print(f"📄 Doc: {doc_info.get('chunks_created', 0)} chunks, {doc_info.get('pages_processed', 0)} pages, Cached: {doc_info.get('cached', False)}")
                
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            if 'answers' in result:
                print(f"📝 Error answers:")
                for i, answer in enumerate(result['answers'], 1):
                    print(f"   {i}. {answer}")
        
        # Log to file (keep detailed logging in file)
        logger.info(f"RESPONSE - Test {test_num}")
        logger.info(f"Duration: {duration:.2f}s, Success: {result['success']}")
        if result['success']:
            logger.info(f"Answers: {json.dumps(result['answers'], indent=2)}")
        else:
            logger.error(f"Error: {result.get('error', 'Unknown error')}")

    async def run_single_test(self, test_case, test_num, total_tests):
        """Run a single test case with detailed logging"""
        self.print_test_header(test_case['name'], test_num, total_tests)
        
        # Print detailed request
        self.print_detailed_request(test_case, test_num)
        
        print(f"\n🚀 PROCESSING...")
        logger.info(f"Starting processing for test {test_num}")
        
        start_time = time.time()
        try:
            result = await process_document_and_answer(
                test_case['document_url'],
                test_case['questions']
            )
            duration = time.time() - start_time
            
            # Print detailed response
            self.print_detailed_response(result, duration, test_num)
            
            logger.info(f"Test {test_num} completed successfully in {duration:.2f}s")
            return True, result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ TEST FAILED after {duration:.2f} seconds")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            print(f"{'='*60}")
            
            logger.error(f"Test {test_num} failed after {duration:.2f}s: {str(e)}")
            return False, {"success": False, "error": str(e)}

    async def run_selected_tests(self, test_numbers=None):
        """Run selected test cases or all if none specified"""
        if test_numbers:
            # Validate test numbers
            invalid_nums = [num for num in test_numbers if num < 1 or num > len(self.test_cases)]
            if invalid_nums:
                print(f"❌ Invalid test numbers: {invalid_nums}")
                print(f"Available test numbers: 1-{len(self.test_cases)}")
                return []
            
            selected_tests = [(i-1, self.test_cases[i-1]) for i in test_numbers]
            suite_type = f"Selected Tests ({', '.join(map(str, test_numbers))})"
        else:
            selected_tests = list(enumerate(self.test_cases))
            suite_type = "All Tests"
        
        start_timestamp = datetime.now()
        print(f"🎯 Policy RAG System - Detailed Test Suite ({suite_type})")
        print(f"📅 Started at: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Running {len(selected_tests)} of {len(self.test_cases)} test cases")
        print(f"📝 Logs saved to: test_results.log")
        self.print_separator()
        
        logger.info("=" * 80)
        logger.info("STARTING POLICY RAG SYSTEM TEST SUITE")
        logger.info(f"Start time: {start_timestamp.isoformat()}")
        logger.info(f"Total test cases: {len(self.test_cases)}")
        logger.info("=" * 80)
        
        results = []
        passed = 0
        failed = 0
        total_start_time = time.time()
        
        for idx, (original_idx, test_case) in enumerate(selected_tests, 1):
            test_num = original_idx + 1  # Display original test number
            success, result = await self.run_single_test(test_case, test_num, len(selected_tests))
            results.append({
                'test_number': test_num,
                'name': test_case['name'],
                'success': success,
                'result': result
            })
            
            if success:
                passed += 1
            else:
                failed += 1
            
            # Add small delay between tests
            if idx < len(selected_tests):
                print(f"\n⏳ Waiting 2 seconds before next test...\n")
                await asyncio.sleep(2)
        
        # Final summary
        end_timestamp = datetime.now()
        total_duration = time.time() - total_start_time
        
        self.print_separator("=")
        print("📈 FINAL SUMMARY")
        self.print_separator("-")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {(passed/len(selected_tests)*100):.1f}%")
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"⏰ Started at: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Completed at: {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 Detailed logs saved to: test_results.log")
        self.print_separator("=")
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("FINAL TEST SUITE SUMMARY")
        logger.info(f"Passed: {passed}, Failed: {failed}")
        logger.info(f"Success Rate: {(passed/len(selected_tests)*100):.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"End time: {end_timestamp.isoformat()}")
        logger.info("=" * 80)
        
        return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Policy RAG System Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_suite.py              # Run all tests
  python test_suite.py 1            # Run only test 1
  python test_suite.py 1 3 5        # Run tests 1, 3, and 5
  python test_suite.py --list       # List all available tests
        """
    )
    
    parser.add_argument(
        'tests', 
        nargs='*', 
        type=int, 
        help='Test numbers to run (1-6). If none specified, runs all tests.'
    )
    
    parser.add_argument(
        '--list', 
        action='store_true', 
        help='List all available tests and exit'
    )
    
    return parser.parse_args()

def list_tests():
    """List all available tests"""
    runner = TestRunner()
    print("📋 Available Tests:")
    print("=" * 60)
    for i, test_case in enumerate(runner.test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Questions: {len(test_case['questions'])}")
        print(f"   Document: {test_case['document_url'][:50]}...")
        print()

def main():
    """Main function to run the test suite"""
    args = parse_arguments()
    
    if args.list:
        list_tests()
        return
    
    try:
        runner = TestRunner()
        
        if args.tests:
            print(f"🎯 Running specific tests: {args.tests}")
            results = asyncio.run(runner.run_selected_tests(args.tests))
        else:
            print("🎯 Running all tests")
            results = asyncio.run(runner.run_selected_tests())
            
        return results
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        logger.warning("Test suite interrupted by user")
        return None
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {str(e)}")
        logger.error(f"Test suite failed: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting Policy RAG System Detailed Test Suite...")
    results = main()
    if results:
        print("✅ Test suite completed successfully!")
    else:
        print("❌ Test suite failed or was interrupted!")
