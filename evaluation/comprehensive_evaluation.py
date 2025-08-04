# comprehensive_invoice_evaluation.py
import json
import requests
import time
import pandas as pd
import re
from typing import List, Dict

class InvoiceRAGEvaluator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
        self.test_cases = [
            # Exact order ID queries
            {
                "query": "Show me invoice with Order ID 10250",
                "relevant_ids": ["invoice_10250"],  # Use order_id format
                "expected_entities": {"order_id": "10250", "customer": "Mario Pontes", "amount": "1813"},
                "query_type": "exact_match"
            },
            {
                "query": "Get invoice for order 10252",
                "relevant_ids": ["invoice_10252"],
                "expected_entities": {"order_id": "10252", "customer": "Pascale Cartrain", "amount": "3730"},
                "query_type": "exact_match"
            },
            
            # Customer queries
            {
                "query": "What invoices are for Mario Pontes?",
                "relevant_ids": ["invoice_10250", "invoice_10253"],  # Mario has orders 10250 and 10253
                "expected_entities": {"customer": "Mario Pontes"},
                "query_type": "customer_lookup"
            },
            {
                "query": "Show me VICTE customer orders",
                "relevant_ids": ["invoice_10251"],
                "expected_entities": {"customer": "Mary Saveley", "customer_id": "VICTE"},
                "query_type": "customer_lookup"
            },
            
            # Location-based queries
            {
                "query": "Which invoices are from Brazil?",
                "relevant_ids": ["invoice_10250", "invoice_10253", "invoice_10256", "invoice_10261"],
                "expected_entities": {"country": "Brazil"},
                "query_type": "location_filter"
            },
            
            # Amount-based queries
            {
                "query": "Which invoice has total price 2490.5?",
                "relevant_ids": ["invoice_10255"],
                "expected_entities": {"amount": "2490.5", "customer": "Michael Holz"},
                "query_type": "amount_lookup"
            },
            
            # Date-based queries
            {
                "query": "Show invoices from July 15, 2016",
                "relevant_ids": ["invoice_10256"],
                "expected_entities": {"date": "2016-07-15", "customer": "Paula Parente"},
                "query_type": "date_filter"
            }
        ]
        
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite"""
        print("🚀 Starting Invoice RAG Evaluation...")
        
        # Run evaluations
        retrieval_metrics = self.evaluate_retrieval()
        answer_metrics = self.evaluate_answer_quality()
        performance_metrics = self.evaluate_performance()
        strategy_metrics = self.evaluate_strategy_routing()
        
        # Combine results
        results = {
            "retrieval": retrieval_metrics,
            "answer_quality": answer_metrics,
            "performance": performance_metrics,
            "strategy": strategy_metrics,
            "overall_score": self._calculate_overall_score(retrieval_metrics, answer_metrics)
        }
        
        self._print_results(results)
        return results
    
    def evaluate_retrieval(self) -> Dict:
        """Evaluate retrieval quality"""
        print("📊 Evaluating retrieval quality...")
        
        precisions, recalls, mrrs = [], [], []
        
        for case in self.test_cases:
            response = self._get_rag_response(case["query"])
            
            if response:
                retrieved_ids = self._extract_retrieved_ids(response)
                relevant_set = set(case["relevant_ids"])
                retrieved_set = set(retrieved_ids[:10])
                
                # Handle empty retrieved_ids
                if not retrieved_ids:
                    precisions.append(0.0)
                    recalls.append(0.0)
                    mrrs.append(0.0)
                    continue
                
                # Precision@5
                precision = len(relevant_set & set(retrieved_ids[:5])) / min(5, len(retrieved_ids))
                precisions.append(precision)
                
                # Recall@10
                recall = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0
                recalls.append(recall)
                
                # MRR
                mrr = 0
                for i, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_set:
                        mrr = 1/i
                        break
                mrrs.append(mrr)
            else:
                # No response received
                precisions.append(0.0)
                recalls.append(0.0)
                mrrs.append(0.0)
        
        return {
            "precision@5": sum(precisions) / len(precisions) if precisions else 0,
            "recall@10": sum(recalls) / len(recalls) if recalls else 0,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else 0
        }
        
        return {
            "precision@5": sum(precisions) / len(precisions),
            "recall@10": sum(recalls) / len(recalls),
            "mrr": sum(mrrs) / len(mrrs)
        }
    
    def evaluate_answer_quality(self) -> Dict:
        """Evaluate answer quality"""
        print("✅ Evaluating answer quality...")
        
        entity_accuracies = []
        factual_accuracies = []
        
        for case in self.test_cases:
            response = self._get_rag_response(case["query"])
            
            if response:
                answer = response.get("answer", "")
                
                # Entity extraction accuracy
                entity_acc = self._calculate_entity_accuracy(answer, case.get("expected_entities", {}))
                entity_accuracies.append(entity_acc)
                
                # Factual accuracy (basic check)
                factual_acc = self._calculate_factual_accuracy(answer, case)
                factual_accuracies.append(factual_acc)
            else:
                # No response - score as 0
                entity_accuracies.append(0.0)
                factual_accuracies.append(0.0)
        
        return {
            "entity_accuracy": sum(entity_accuracies) / len(entity_accuracies) if entity_accuracies else 0,
            "factual_accuracy": sum(factual_accuracies) / len(factual_accuracies) if factual_accuracies else 0
        }
    
    def evaluate_performance(self) -> Dict:
        """Evaluate performance metrics"""
        print("⚡ Evaluating performance...")
        
        response_times = []
        
        for case in self.test_cases:
            start_time = time.time()
            response = self._get_rag_response(case["query"])
            response_time = time.time() - start_time
            
            response_times.append(response_time)
        
        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "max_response_time": max(response_times),
            "min_response_time": min(response_times)
        }
    
    def evaluate_strategy_routing(self) -> Dict:
        """Evaluate if correct strategy (SQL vs Vector) was chosen"""
        print("🎯 Evaluating strategy routing...")
        
        correct_strategies = 0
        total_cases = 0
        
        for case in self.test_cases:
            response = self._get_rag_response(case["query"])
            
            if response:
                thinking = response.get("thinking", [])
                strategy_used = "unknown"
                
                for step in thinking:
                    if "direct_sql" in step.lower():
                        strategy_used = "direct_sql"
                    elif "vector_search" in step.lower():
                        strategy_used = "vector_search"
                
                # Expected strategy based on query type
                expected_strategy = "direct_sql" if case["query_type"] in ["exact_match", "customer_lookup", "amount_lookup", "date_filter"] else "vector_search"
                
                if strategy_used == expected_strategy:
                    correct_strategies += 1
                
                total_cases += 1
        
        return {
            "strategy_accuracy": correct_strategies / total_cases if total_cases > 0 else 0
        }
    
    def _get_rag_response(self, query: str) -> Dict:
        """Get response from RAG API with better error handling"""
        try:
            print(f"  Testing: {query}")
            response = requests.post(f"{self.api_url}/rag/ask", 
                                params={"question": query, "user_id": 1},
                                timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"    ❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"    ❌ Request failed: {e}")
            return None
    
    def _extract_retrieved_ids(self, response: Dict) -> List[str]:
        """Extract retrieved document IDs from BOTH vector and SQL sources"""
        retrieved_ids = []
        
        sources = response.get("sources", {})
        
        # Check vector search results
        vector_search = sources.get("vector_search", [])
        for source in vector_search:
            if source.get("type") == "invoice":
                order_id = source.get("order_id")
                if order_id:
                    retrieved_ids.append(f"invoice_{order_id}")
            elif source.get("type") == "line_item":
                order_id = source.get("order_id")
                rank = source.get("rank", 1)
                if order_id:
                    retrieved_ids.append(f"item_{order_id}_{rank-1}")
        
        # If no vector results, check SQL results (for direct SQL queries)
        if not retrieved_ids:
            sql_results = sources.get("sql_results", [])
            if sql_results:
                # Extract order_id from SQL results
                for row in sql_results:
                    order_id = row.get("order_id")
                    if order_id:
                        retrieved_ids.append(f"invoice_{order_id}")
                        break  # Only need one invoice ID for exact matches
        
        return retrieved_ids
    
    def _calculate_entity_accuracy(self, answer: str, expected_entities: Dict) -> float:
        """Calculate entity extraction accuracy"""
        if not expected_entities:
            return 1.0
        
        correct = 0
        total = len(expected_entities)
        
        for entity_type, expected_value in expected_entities.items():
            if entity_type == "order_id":
                if expected_value in answer:
                    correct += 1
            elif entity_type == "amount":
                if expected_value in answer or f"${expected_value}" in answer:
                    correct += 1
            elif entity_type == "customer":
                if expected_value.lower() in answer.lower():
                    correct += 1
            elif entity_type in ["city", "country", "date"]:
                if expected_value.lower() in answer.lower():
                    correct += 1
        
        return correct / total
    
    def _calculate_factual_accuracy(self, answer: str, case: Dict) -> float:
        """Basic factual accuracy check"""
        # Simple check - if answer contains expected entities, it's likely factually correct
        expected_entities = case.get("expected_entities", {})
        if not expected_entities:
            return 1.0
        
        return self._calculate_entity_accuracy(answer, expected_entities)
    
    def _calculate_overall_score(self, retrieval: Dict, answer_quality: Dict) -> float:
        """Calculate weighted overall score"""
        return (
            retrieval["precision@5"] * 0.3 +
            retrieval["recall@10"] * 0.2 +
            retrieval["mrr"] * 0.2 +
            answer_quality["entity_accuracy"] * 0.2 +
            answer_quality["factual_accuracy"] * 0.1
        )
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*60)
        print("📊 INVOICE RAG EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n🔍 RETRIEVAL METRICS:")
        print(f"  Precision@5:     {results['retrieval']['precision@5']:.3f}")
        print(f"  Recall@10:       {results['retrieval']['recall@10']:.3f}")
        print(f"  MRR:             {results['retrieval']['mrr']:.3f}")
        
        print(f"\n✅ ANSWER QUALITY:")
        print(f"  Entity Accuracy: {results['answer_quality']['entity_accuracy']:.3f}")
        print(f"  Factual Accuracy:{results['answer_quality']['factual_accuracy']:.3f}")
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Avg Response:    {results['performance']['avg_response_time']:.2f}s")
        print(f"  Strategy Acc:    {results['strategy']['strategy_accuracy']:.3f}")
        
        print(f"\n🎯 OVERALL SCORE:   {results['overall_score']:.3f}")
        print("="*60)

if __name__ == "__main__":
    evaluator = InvoiceRAGEvaluator()
    results = evaluator.run_full_evaluation()
    
    # Save results
    with open("invoice_rag_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to 'invoice_rag_evaluation.json'")
