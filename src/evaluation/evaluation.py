import json
import logging
import pathlib
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import os
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from src.generation.generation import ask
from src.retrieval.indexing import load_indexes
from src.retrieval.retrieval import EnhancedRetriever, HybridRetriever, RerankPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET = [
    # EASY / DIRECT (1-10)
    {"id": 1, "question": "What is the SI unit of speed?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 2, "question": "Define acceleration.", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 3, "question": "What is Newton's first law of motion?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 4, "question": "What is the formula for momentum?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 5, "question": "What is the value of the universal gravitational constant G?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 6, "question": "State the law of conservation of momentum.", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 7, "question": "What is balanced force?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 8, "question": "Define displacement.", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 9, "question": "What does the slope of a distance-time graph represent?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    {"id": 10, "question": "What is weight?", "difficulty": "easy", "type": "direct", "expected_oos": False},
    
    # MEDIUM / VARIED (11-25)
    {"id": 11, "question": "Explain why a person falls forward when a moving bus brakes suddenly.", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 12, "question": "An object travels 16 m in 4 s and then another 16 m in 2 s. What is the average speed of the object?", "difficulty": "medium", "type": "worked_example", "expected_oos": False},
    {"id": 13, "question": "Compare the inertia of a rubber ball and a stone of the same size.", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 14, "question": "What happens to the force between two objects if the mass of one object is doubled?", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 15, "question": "Calculate the force required to impart a velocity of 30 m/s in 10 s to a car of mass 1500 kg.", "difficulty": "medium", "type": "worked_example", "expected_oos": False},
    {"id": 16, "question": "Explain the difference between mass and weight.", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 17, "question": "How does the force of gravitation between two objects change when the distance between them is reduced to half?", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 18, "question": "Why is the weight of an object on the moon 1/6th its weight on the earth?", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 19, "question": "A stone is released from the top of a tower of height 19.6 m. Calculate its final velocity just before touching the ground.", "difficulty": "medium", "type": "worked_example", "expected_oos": False},
    {"id": 20, "question": "What is the importance of the universal law of gravitation?", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 21, "question": "Distinguish between uniform and non-uniform motion.", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 22, "question": "A motorboat starting from rest on a lake accelerates in a straight line at a constant rate of 3.0 m/s^2 for 8.0 s. How far does the boat travel during this time?", "difficulty": "medium", "type": "worked_example", "expected_oos": False},
    {"id": 23, "question": "Explain Newton's third law with an example from the text.", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 24, "question": "Why do we use seat belts in cars?", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    {"id": 25, "question": "What is free fall?", "difficulty": "medium", "type": "paraphrased", "expected_oos": False},
    
    # HARD (26-30)
    {"id": 26, "question": "How does the velocity of a falling object change if both the height of release and the gravitational constant were doubled? (Hypothetical Multi-hop)", "difficulty": "hard", "type": "multi_hop", "expected_oos": False},
    {"id": 27, "question": "Derive the relationship between force, mass and acceleration using the second law of motion.", "difficulty": "hard", "type": "multi_hop", "expected_oos": False},
    {"id": 28, "question": "Calculate the gravitational force between a person on Mars and the Martian surface using Earth's radius.", "difficulty": "hard", "type": "oos_plausible", "expected_oos": True},
    {"id": 29, "question": "Who invented the first steam engine mentioned in the Motion chapter?", "difficulty": "hard", "type": "oos_clear", "expected_oos": True},
    {"id": 30, "question": "What is the chemical composition of the atmosphere on Jupiter according to the Gravitation chapter?", "difficulty": "hard", "type": "oos_clear", "expected_oos": True}
]

def score_response(response: dict, item: dict) -> dict:
    """Interactively scores a response against a golden set item."""
    # Correctness
    while True:
        val = input(f"[{item['id']}] Correctness (Y/N/P): ").strip().upper()
        if val in ["Y", "N", "P"]:
            correctness = val
            break
        print("Invalid input. Use Y, N, or P.")
        
    # Grounding
    if not response['chunk_ids']:
        print("No citations — auto-grounding: N")
        grounding = "N"
    else:
        while True:
            val = input("Citations valid? (Y/N): ").strip().upper()
            if val in ["Y", "N"]:
                grounding = val
                break
            print("Invalid input. Use Y or N.")
            
    # Refusal
    if not item['expected_oos']:
        refusal = "NA"
    else:
        # auto-detect
        ans_low = response['answer'].lower()
        is_refusal = "don't have" in ans_low or "not covered" in ans_low
        refusal = "Y" if is_refusal else "N"
        
    return {
        "id": item['id'],
        "correctness": correctness,
        "grounding": grounding,
        "refusal": refusal
    }

def run_evaluation(golden_set: list, ask_fn, retriever, output_csv: str) -> pd.DataFrame:
    """Runs end-to-end evaluation on the golden set."""
    records = []
    for item in golden_set:
        print(f"\nEvaluating ID {item['id']}: {item['question']}")
        response = ask_fn(item['question'], retriever)
        
        print(f"Answer: {response['answer'][:300]}...")
        print(f"Citations: {response['chunk_ids']}")
        
        scores = score_response(response, item)
        
        record = {
            **item,
            "answer": response['answer'],
            "chunk_ids": response['chunk_ids'],
            "correctness": scores['correctness'],
            "grounding": scores['grounding'],
            "refusal": scores['refusal']
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    # Summary
    n_correct = len(df[df['correctness'] == 'Y'])
    n_grounded = len(df[df['grounding'] == 'Y'])
    oos_df = df[df['expected_oos'] == True]
    n_refused = len(oos_df[oos_df['refusal'] == 'Y'])
    n_oos = len(oos_df)
    
    print(f"\nEvaluation Summary:")
    print(f"  Correctness: {n_correct}/30 ({n_correct/30*100:.1f}%)")
    print(f"  Grounding: {n_grounded}/30 ({n_grounded/30*100:.1f}%)")
    if n_oos > 0:
        print(f"  Refusal: {n_refused}/{n_oos} ({n_refused/n_oos*100:.1f}%)")
        
    return df

def compute_delta(df_v1: pd.DataFrame, df_v2: pd.DataFrame) -> dict:
    """Computes changes in metrics between two evaluation runs."""
    metrics = {}
    for col in ['correctness', 'grounding', 'refusal']:
        v1_count = len(df_v1[df_v1[col] == 'Y'])
        v2_count = len(df_v2[df_v2[col] == 'Y'])
        metrics[f"{col}_delta"] = v2_count - v1_count
        
    improved = []
    regressed = []
    
    for i in range(len(df_v1)):
        id_val = df_v1.iloc[i]['id']
        v1_corr = df_v1.iloc[i]['correctness']
        v2_corr = df_v2[df_v2.id == id_val].iloc[0]['correctness']
        
        if v1_corr in ['N', 'P'] and v2_corr == 'Y':
            improved.append(id_val)
        elif v1_corr == 'Y' and v2_corr in ['N', 'P']:
            regressed.append(id_val)
            
    metrics['improved_ids'] = improved
    metrics['regressed_ids'] = regressed
    return metrics

def run_ragas(eval_df: pd.DataFrame, retriever, judge_model: str = "claude-3-haiku-20240307") -> pd.DataFrame:
    """Runs RAGAS evaluation on the results."""
    # Build dataset
    data = {
        "question": eval_df["question"].tolist(),
        "answer": eval_df["answer"].tolist(),
        "contexts": [[c["content"] for c in ask(q, retriever)["chunks_used"]] for q in eval_df["question"]],
        "ground_truth": ["" for _ in range(len(eval_df))] # No ground truth provided in Golden Set
    }
    
    dataset = Dataset.from_dict(data)
    
    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model=judge_model, temperature=0)
    else:
        logger.info("ANTHROPIC_API_KEY missing, using ChatGroq for evaluation")
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # Ragas evaluate usually needs an LLM wrapper. 
    # For now we'll call evaluate directly assuming environment is set.
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm
    )
    
    res_df = result.to_pandas()
    res_df.to_csv("./outputs/ragas_report.csv", index=False)
    
    print("\nRAGAS Metrics:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
        
    if result["faithfulness"] < 0.7:
        print("WARNING: Check context_precision and recall")
        
    return res_df

if __name__ == "__main__":
    chunks_path = "./data/processed/chunks.json"
    persist_dir = "./vectorstore/chroma_db"
    collection_name = "parishiksha_v2"
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    vs, bm25 = load_indexes(persist_dir, collection_name, chunks)
    hybrid = HybridRetriever(vs, bm25, chunks)
    reranker = RerankPipeline(use_cohere=True)
    retriever = EnhancedRetriever(hybrid, reranker, use_multiquery=True)
    
    # Run evaluation
    df_scored = run_evaluation(GOLDEN_SET, ask, retriever, "eval_scored.csv")
    
    print("Evaluation complete. Review eval_scored.csv before proceeding to Step 9.")
