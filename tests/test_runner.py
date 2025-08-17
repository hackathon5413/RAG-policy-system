#!/usr/bin/env python3
"""
Clean Test Runner for Policy RAG System
Runs test cases with LLM-based evaluation and accuracy scoring
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change to project root directory so relative paths work
os.chdir(str(project_root))

from app.document_processor import process_document_and_answer
from tests.evaluator import AnswerEvaluator
from tests.data.test_cases import TEST_CASES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "test_results.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("TestRunner")

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app.embeddings").setLevel(logging.WARNING)


class TestRunner:
    """Main test runner class"""

    def __init__(self):
        self.evaluator = AnswerEvaluator()
        self.test_cases = TEST_CASES
        self.results_dir = Path(__file__).parent / "data" / "results"
        self.results_dir.mkdir(exist_ok=True)

    def save_test_results(self, results: list[dict[str, Any]], start_timestamp: datetime, end_timestamp: datetime, test_numbers: list[int] | None = None) -> str:
        """Save complete test run results to a timestamped JSON file"""
        timestamp_str = start_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"test_run_{timestamp_str}.json"
        filepath = self.results_dir / filename

        # Calculate summary statistics
        total_tests = len(results)
        passed = sum(1 for r in results if r.get("success", False))
        failed = total_tests - passed

        # Calculate evaluation statistics if available
        evaluated_results = [r for r in results if r.get("evaluation")]
        avg_accuracy = 0
        avg_intent_rate = 0
        if evaluated_results:
            accuracies = [r["evaluation"]["summary"]["average_accuracy"] for r in evaluated_results]
            intent_rates = [r["evaluation"]["summary"]["intent_fulfillment_rate"] for r in evaluated_results]
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_intent_rate = sum(intent_rates) / len(intent_rates)

        # Create comprehensive test run data
        test_run_data = {
            "metadata": {
                "timestamp": start_timestamp.isoformat(),
                "end_timestamp": end_timestamp.isoformat(),
                "duration_seconds": (end_timestamp - start_timestamp).total_seconds(),
                "test_framework_version": "2.0",
                "total_tests_available": len(self.test_cases),
                "tests_executed": total_tests,
                "test_numbers_run": test_numbers or list(range(1, len(self.test_cases) + 1))
            },
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "success_rate_percent": (passed / total_tests * 100) if total_tests > 0 else 0,
                "evaluated_tests": len(evaluated_results),
                "average_accuracy_percent": avg_accuracy,
                "average_intent_fulfillment_percent": avg_intent_rate
            },
            "results": results
        }

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_run_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Test results saved to: {filepath}")
            print(f"💾 Test results saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            print(f"❌ Failed to save test results: {e}")
            return ""

    def print_separator(self, char="=", length=80):
        """Print a separator line"""
        print(char * length)

    def print_test_header(self, test_name: str, test_num: int, total_tests: int):
        """Print minimal test header"""
        print(f"\n🧪 Test {test_num}/{total_tests}: {test_name}")

    def print_test_details(self, test_case: dict[str, Any], test_num: int):
        """Print minimal test details"""
        questions = test_case["questions"]
        print(f"� Test {test_num}: {len(questions)} questions")

        logger.info(f"REQUEST - Test {test_num}")
        logger.info(f"Document URL: {test_case['document_url']}")
        logger.info(f"Questions ({len(questions)}): {json.dumps(questions, indent=2)}")

    def print_response_details(self, result: dict[str, Any], duration: float, test_num: int):
        """Print minimal response details"""
        success = result.get("success", False)
        answers = result.get("answers", [])

        status = "✅ Success" if success else "❌ Failed"
        print(f"� {status}: {duration:.1f}s | {len(answers)} answers")

        logger.info(f"RESPONSE - Test {test_num}")
        logger.info(f"Duration: {duration:.2f}s, Success: {success}")
        logger.info(f"Answers: {json.dumps(answers, indent=2)}")

    async def print_evaluation_results(self, evaluation_result: dict[str, Any], test_num: int):
        """Print minimal evaluation results"""
        summary = evaluation_result["summary"]

        avg_accuracy = summary.get('average_accuracy', 0)
        intent_rate = summary.get('intent_fulfillment_rate', 0)
        total_q = summary.get('total_questions', 0)

        print(f"� Evaluation: {avg_accuracy:.1f}% accuracy, {intent_rate:.1f}% intent ({total_q}Q)")

        logger.info(f"EVALUATION - Test {test_num}")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    async def run_single_test(self, test_case: dict[str, Any], test_num: int, total_tests: int) -> dict[str, Any]:
        """Run a single test case with evaluation"""
        self.print_test_header(test_case["name"], test_num, total_tests)
        self.print_test_details(test_case, test_num)

        logger.info(f"Starting processing for test {test_num}")

        start_time = time.time()
        try:
            # Run the test
            result = await process_document_and_answer(
                test_case["document_url"],
                test_case["questions"]
            )
            duration = time.time() - start_time

            # Print response details
            self.print_response_details(result, duration, test_num)

            # Evaluate answers if we have expected answers
            if result["success"] and "expected_answers" in test_case:
                evaluation_result = await self.evaluator.evaluate_multiple_answers(
                    test_case["questions"],
                    test_case["expected_answers"],
                    result["answers"]
                )
                await self.print_evaluation_results(evaluation_result, test_num)

                return {
                    "success": True,
                    "result": result,
                    "evaluation": evaluation_result,
                    "duration": duration
                }
            else:
                print("\n⏭️ SKIPPING EVALUATION (No expected answers provided)")
                return {
                    "success": True,
                    "result": result,
                    "evaluation": None,
                    "duration": duration
                }

        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ TEST FAILED after {duration:.2f} seconds")
            print(f"{'=' * 60}")
            print(f"Error: {e!s}")
            print(f"{'=' * 60}")

            logger.error(f"Test {test_num} failed after {duration:.2f}s: {e!s}")
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "evaluation": None
            }

    async def run_selected_tests(self, test_numbers: list[int] | None = None) -> list[dict[str, Any]]:
        """Run selected test cases or all if none specified"""
        if test_numbers:
            # Validate test numbers
            invalid_nums = [num for num in test_numbers if num < 1 or num > len(self.test_cases)]
            if invalid_nums:
                print(f"❌ Invalid test numbers: {invalid_nums}")
                print(f"Available test numbers: 1-{len(self.test_cases)}")
                return []

            selected_tests = [(i - 1, self.test_cases[i - 1]) for i in test_numbers]
        else:
            selected_tests = list(enumerate(self.test_cases))

        start_timestamp = datetime.now()
        print(f"🎯 Running {len(selected_tests)} test(s)")

        logger.info("STARTING ENHANCED POLICY RAG SYSTEM TEST SUITE")
        logger.info(f"Start time: {start_timestamp.isoformat()}")
        logger.info(f"Total test cases: {len(self.test_cases)}")
        logger.info("=" * 80)

        results = []
        total_start_time = time.time()

        for _idx, (original_idx, test_case) in enumerate(selected_tests, 1):
            test_num = original_idx + 1  # Display original test number
            test_result = await self.run_single_test(test_case, test_num, len(selected_tests))

            test_result["test_number"] = test_num
            test_result["name"] = test_case["name"]
            results.append(test_result)

        # Final summary
        end_timestamp = datetime.now()
        saved_file = ""
        if results:
            saved_file = self.save_test_results(
                results=results,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                test_numbers=test_numbers
            )

        await self.print_final_summary(results, start_timestamp, total_start_time, saved_file)

        return results

    async def print_final_summary(self, results: list[dict[str, Any]], start_timestamp: datetime, total_start_time: float, saved_file: str = ""):
        """Print final test suite summary"""
        end_timestamp = datetime.now()
        total_duration = time.time() - total_start_time

        # Calculate overall statistics
        passed = sum(1 for r in results if r["success"])
        failed = len(results) - passed

        # Calculate evaluation statistics
        evaluated_results = [r for r in results if r.get("evaluation")]
        if evaluated_results:
            total_accuracy = sum(r["evaluation"]["summary"]["average_accuracy"] for r in evaluated_results)
            avg_accuracy = total_accuracy / len(evaluated_results)
            total_intent_rate = sum(r["evaluation"]["summary"]["intent_fulfillment_rate"] for r in evaluated_results)
            avg_intent_rate = total_intent_rate / len(evaluated_results)
        else:
            avg_accuracy = 0
            avg_intent_rate = 0

        print(f"\n📈 SUMMARY: {passed}/{len(results)} passed ({(passed / len(results) * 100):.1f}%)")

        if evaluated_results:
            print(f"📊 Avg Accuracy: {avg_accuracy:.1f}% | Intent: {avg_intent_rate:.1f}%")

        print(f"⏱️  Duration: {total_duration:.1f}s")

        if saved_file:
            filename = saved_file.split('/')[-1] if '/' in saved_file else saved_file
            print(f"📁 Results: {filename}")

        logger.info("FINAL TEST SUITE SUMMARY")
        logger.info(f"Passed: {passed}, Failed: {failed}")
        logger.info(f"Success Rate: {(passed / len(results) * 100):.1f}%")
        if evaluated_results:
            logger.info(f"Average Accuracy: {avg_accuracy:.1f}%")
            logger.info(f"Average Intent Fulfillment: {avg_intent_rate:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"End time: {end_timestamp.isoformat()}")
        logger.info(f"Passed: {passed}, Failed: {failed}")
        logger.info(f"Success Rate: {(passed / len(results) * 100):.1f}%")
        if evaluated_results:
            logger.info(f"Average Accuracy: {avg_accuracy:.1f}%")
            logger.info(f"Average Intent Fulfillment: {avg_intent_rate:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"End time: {end_timestamp.isoformat()}")
        logger.info("=" * 80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Policy RAG System Test Suite with LLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_runner.py              # Run all tests
  python test_runner.py 1            # Run only test 1
  python test_runner.py 1 3 5        # Run tests 1, 3, and 5
  python test_runner.py --list       # List all available tests
        """,
    )

    parser.add_argument(
        "tests",
        nargs="*",
        type=int,
        help="Test numbers to run. If none specified, runs all tests.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tests and exit"
    )

    return parser.parse_args()


def list_tests():
    """List all available tests"""
    print("📋 Available Tests:")
    print("=" * 60)
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Questions: {len(test_case['questions'])}")
        print(f"   Expected Answers: {len([a for a in test_case.get('expected_answers', []) if a.strip()])}")
        print(f"   Document: {test_case['document_url'][:50]}...")
        print()


async def main():
    """Main function to run the test suite"""
    args = parse_arguments()

    if args.list:
        list_tests()
        return

    try:
        runner = TestRunner()

        if args.tests:
            print(f"🎯 Running specific tests: {args.tests}")
            results = await runner.run_selected_tests(args.tests)
        else:
            print("🎯 Running all tests")
            results = await runner.run_selected_tests()

        return results
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        logger.warning("Test suite interrupted by user")
        return None
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e!s}")
        logger.error(f"Test suite failed: {e!s}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Enhanced Policy RAG System Test Suite...")
    results = asyncio.run(main())
    if results:
        print("✅ Test suite completed successfully!")
    else:
        print("❌ Test suite failed or was interrupted!")
