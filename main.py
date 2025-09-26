"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain with multi-judge evaluation.
Enhanced with advanced similarity search and comprehensive metrics tracking.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Suppress torch warning
warnings.filterwarnings('ignore', category = UserWarning, message = '.*torch.classes.*')

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")
METRICS_DIR = os.path.join("data", "metrics")

# Ensure directories exist
os.makedirs(METRICS_DIR, exist_ok = True)
os.makedirs(PERSIST_DIRECTORY, exist_ok = True)

# Streamlit page configuration
st.set_page_config(
    page_title = "DeepLaw",
    page_icon = "🧠",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

# Logging configuration
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@dataclass
class LLMEvaluation:
    """Comprehensive LLM evaluation metrics using LLM-as-a-judge"""
    faithfulness: float  # 0-10: Does the answer rely on the provided context?
    groundedness: float  # 0-10: Can information be traced back to context?
    factual_consistency: float  # 0-10: Factual alignment with context
    relevance: float  # 0-10: Addresses the actual query
    completeness: float  # 0-10: Covers all important aspects
    fluency: float  # 0-10: Natural, coherent, and well-written
    overall_score: float  # 0-10: Overall quality score
    evaluation_notes: str  # Detailed explanation from judge
    judge_model: str  # Which model performed this evaluation

@dataclass
class LLMMetrics:
    """Data class to store LLM performance metrics"""
    timestamp: str
    query: str
    response: str
    context: str
    response_time: float
    token_count: int
    tokens_per_second: float
    model: str
    session_id: str
    evaluations: List[LLMEvaluation]  # Multiple evaluations from different judges

class LLMJudgeEvaluator:
    """LLM-as-a-judge evaluation system using multiple models"""
    
    def __init__(self, judge_models):
        self.judge_models = judge_models
        self.evaluation_prompt = """You are an expert evaluator of AI responses. Please evaluate the following response based on
        the given context and query.

        QUERY: {query}

        CONTEXT: {context}

        RESPONSE: {response}

        Please evaluate on a scale of 0.0-10.0 for each criterion:

        1. FAITHFULNESS (0.0-10.0): Does the answer rely solely on the provided context without hallucination?
        2. GROUNDEDNESS (0.0-10.0): Can all information be directly traced back to the context?
        3. FACTUAL CONSISTENCY (0.0-10.0): How factually accurate is the response compared to the context?
        4. RELEVANCE (0.0-10.0): How well does the response address the specific query?
        5. COMPLETENESS (0.0-10.0): Does the response cover all important aspects of the query?
        6. FLUENCY (0.0-10.0): Is the response natural, coherent, and well-written?

        Calculate an overall_score (0.0-10.0) as a weighted average:
        - Faithfulness, Groundedness, Factual Consistency: 20% each
        - Relevance: 15%
        - Completeness: 15%
        - Fluency: 10%

        Provide your evaluation in JSON format exactly as follows:
        {{
        "faithfulness": 8.5,
        "groundedness": 9.0,
        "factual_consistency": 9.2,
        "relevance": 8.8,
        "completeness": 7.5,
        "fluency": 9.5,
        "overall_score": 8.7,
        "evaluation_notes": "Brief explanation of scores"
        }}

        Only respond with valid JSON, no other text."""
    
    def _get_rating_category(self, score: float) -> str:
        """Convert overall score to rating category"""
        if score >= 9.0:
            return "Excellent"
        elif score >= 8.0:
            return "Good"
        elif score >= 6.5:
            return "Fair"
        elif score >= 5.0:
            return "Average"
        else:
            return "Poor / Weak"
    
    def evaluate_response(self, query: str, response: str, context: str) -> List[LLMEvaluation]:
        """Evaluate response using multiple LLM judges"""
        evaluations = []
        
        for judge_model in self.judge_models:
            try:
                judge_llm = ChatOllama(model = judge_model, request_timeout = 3600.0)
                
                prompt = self.evaluation_prompt.format(
                    query = query,
                    context = context[:2000],  # Limit context length for evaluation
                    response = response
                )
                
                evaluation_response = judge_llm.invoke(prompt)
                eval_text = evaluation_response.content.strip()
                eval_data = self._parse_evaluation_response(eval_text)
                
                # Add rating category to evaluation notes
                overall_score = eval_data.get('overall_score', 5.0)
                rating_category = self._get_rating_category(overall_score)
                eval_data['evaluation_notes'] = f"{rating_category}: {eval_data.get('evaluation_notes', '')}"
                eval_data['judge_model'] = judge_model
                
                evaluations.append(LLMEvaluation(**eval_data))
                
            except Exception as e:
                print(f"Evaluation error from {judge_model}: {e}")
                evaluations.append(LLMEvaluation(
                    faithfulness = 5.0,
                    groundedness = 5.0,
                    factual_consistency = 5.0,
                    relevance = 5.0,
                    completeness = 5.0,
                    fluency = 6.0,
                    overall_score = 5.2,
                    evaluation_notes = f"Poor / Weak: Evaluation failed: {str(e)}",
                    judge_model = judge_model
                ))
        
        return evaluations
    
    def _parse_evaluation_response(self, text: str) -> Dict[str, Any]:
        """Parse the evaluation response from the judge LLM"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback evaluation
        return {
            "faithfulness": 5.0,
            "groundedness": 5.0,
            "factual_consistency": 5.0,
            "relevance": 5.0,
            "completeness": 5.0,
            "fluency": 6.0,
            "overall_score": 5.2,
            "evaluation_notes": "Average: Automatic fallback evaluation"
        }

class MetricsCollector:
    """Collects and manages LLM performance metrics with multi-judge evaluation"""
    
    def __init__(self, metrics_dir: str = METRICS_DIR):
        self.metrics_dir = metrics_dir
        self.current_session_metrics: List[LLMMetrics] = []
    
    def record_metrics(self, query: str, response: str, context: str, 
                     response_time: float, token_count: int, 
                     model: str, session_id: str,
                     evaluations: List[LLMEvaluation]) -> LLMMetrics:
        """Record metrics with multi-judge evaluation"""
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        metrics = LLMMetrics(
            timestamp = datetime.now().isoformat(),
            query = query,
            response = response,
            context = context[:1000],  # Limit context length for storage
            response_time = response_time,
            token_count = token_count,
            tokens_per_second = tokens_per_second,
            model = model,
            session_id = session_id,
            evaluations = evaluations
        )
        
        self.current_session_metrics.append(metrics)
        
        # Auto-save each evaluation to JSON file
        self._save_single_evaluation(metrics)
        
        return metrics
    
    def _dataclass_to_dict(self, obj):
        """Safely convert dataclass to dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            # It's a dataclass instance
            result = {}
            for field in obj.__dataclass_fields__:
                value = getattr(obj, field)
                if isinstance(value, list):
                    result[field] = [self._dataclass_to_dict(item) if hasattr(item, '__dataclass_fields__') else item for item in value]
                else:
                    result[field] = self._dataclass_to_dict(value) if hasattr(value, '__dataclass_fields__') else value
            return result
        else:
            # It's a regular Python object
            return obj
    
    def _save_single_evaluation(self, metric: LLMMetrics) -> str:
        """Save a single evaluation to JSON file"""
        filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(metric.query[:20])}.json"
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Use our safe conversion method instead of asdict()
        metric_dict = self._dataclass_to_dict(metric)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metric_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")
            return ""
    
    def save_all_metrics_to_file(self, filename: str = None) -> str:
        """Save all metrics to a single JSON file"""
        if not filename:
            filename = f"llm_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        metrics_dicts = []
        for metric in self.current_session_metrics:
            metric_data = {
                "timestamp": metric.timestamp,
                "query": metric.query,
                "response": metric.response[:2000] + "..." if len(metric.response) > 2000 else metric.response,
                "context_preview": metric.context[:1000] + "..." if len(metric.context) > 1000 else metric.context,
                "response_time": round(metric.response_time, 2),
                "token_count": metric.token_count,
                "tokens_per_second": round(metric.tokens_per_second, 2),
                "model": metric.model,
                "session_id": metric.session_id
            }
            
            if metric.evaluations:
                metric_data["evaluations"] = []
                for eval_obj in metric.evaluations:
                    eval_data = {
                        "faithfulness": round(eval_obj.faithfulness, 1),
                        "groundedness": round(eval_obj.groundedness, 1),
                        "factual_consistency": round(eval_obj.factual_consistency, 1),
                        "relevance": round(eval_obj.relevance, 1),
                        "completeness": round(eval_obj.completeness, 1),
                        "fluency": round(eval_obj.fluency, 1),
                        "overall_score": round(eval_obj.overall_score, 1),
                        "evaluation_notes": eval_obj.evaluation_notes,
                        "judge_model": eval_obj.judge_model
                    }
                    metric_data["evaluations"].append(eval_data)
            
            metrics_dicts.append(metric_data)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_dicts, f, indent=2, ensure_ascii=False)
            logger.info(f"All metrics saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving all metrics: {e}")
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if not self.current_session_metrics:
            return {}
        
        response_times = [m.response_time for m in self.current_session_metrics]
        token_counts = [m.token_count for m in self.current_session_metrics]
        tokens_per_second = [m.tokens_per_second for m in self.current_session_metrics]
        
        all_evaluations = []
        for metric in self.current_session_metrics:
            all_evaluations.extend(metric.evaluations)
        
        summary = {
            "total_interactions": len(self.current_session_metrics),
            "avg_response_time": round(sum(response_times) / len(response_times), 2),
            "min_response_time": round(min(response_times), 2),
            "max_response_time": round(max(response_times), 2),
            "avg_tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 2),
            "total_tokens_generated": sum(token_counts),
            "avg_tokens_per_response": round(sum(token_counts) / len(token_counts), 1),
            "total_evaluations": len(all_evaluations),
            "unique_judges": len(set(eval_obj.judge_model for eval_obj in all_evaluations))
        }
        
        if all_evaluations:
            # Calculate average scores across all evaluations
            summary.update({
                "avg_faithfulness": round(sum(e.faithfulness for e in all_evaluations) / len(all_evaluations), 1),
                "avg_groundedness": round(sum(e.groundedness for e in all_evaluations) / len(all_evaluations), 1),
                "avg_factual_consistency": round(sum(e.factual_consistency for e in all_evaluations) / len(all_evaluations), 1),
                "avg_relevance": round(sum(e.relevance for e in all_evaluations) / len(all_evaluations), 1),
                "avg_completeness": round(sum(e.completeness for e in all_evaluations) / len(all_evaluations), 1),
                "avg_fluency": round(sum(e.fluency for e in all_evaluations) / len(all_evaluations), 1),
                "avg_overall_score": round(sum(e.overall_score for e in all_evaluations) / len(all_evaluations), 1),
            })
            
            # Count rating categories
            rating_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Average": 0, "Poor / Weak": 0}
            for eval_obj in all_evaluations:
                score = eval_obj.overall_score
                if score >= 9.0:
                    rating_counts["Excellent"] += 1
                elif score >= 8.0:
                    rating_counts["Good"] += 1
                elif score >= 6.5:
                    rating_counts["Fair"] += 1
                elif score >= 5.0:
                    rating_counts["Average"] += 1
                else:
                    rating_counts["Poor / Weak"] += 1
            
            summary["rating_distribution"] = rating_counts
            
            # Calculate scores by judge model
            judge_scores = {}
            for eval_obj in all_evaluations:
                if eval_obj.judge_model not in judge_scores:
                    judge_scores[eval_obj.judge_model] = []
                judge_scores[eval_obj.judge_model].append(eval_obj.overall_score)
            
            summary["judge_models"] = {
                judge: {
                    "avg_score": round(sum(scores) / len(scores), 1),
                    "count": len(scores)
                }
                for judge, scores in judge_scores.items()
            }
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        summary = self.get_session_summary()
        if not summary:
            return "No metrics collected yet."
        
        report = [
            "=== MULTI-JUDGE EVALUATION REPORT ===",
            f"Session ID: {self.current_session_metrics[0].session_id if self.current_session_metrics else 'N/A'}",
            f"Total Interactions: {summary['total_interactions']}",
            f"Total Evaluations: {summary['total_evaluations']}",
            f"Unique Judge Models: {summary['unique_judges']}",
            f"Average Response Time: {summary['avg_response_time']}s",
            f"Total Tokens Generated: {summary['total_tokens_generated']}",
            f"Average Throughput: {summary['avg_tokens_per_second']} tokens/s",
        ]
        
        if 'avg_overall_score' in summary:
            report.extend([
                "",
                "=== OVERALL QUALITY EVALUATION (0.0-10.0 scale) ===",
                f"Overall Quality: {summary['avg_overall_score']}/10.0",
                f"Faithfulness: {summary['avg_faithfulness']}/10.0 (reliance on context)",
                f"Groundedness: {summary['avg_groundedness']}/10.0 (traceability to context)",
                f"Factual Consistency: {summary['avg_factual_consistency']}/10.0 (accuracy vs context)",
                f"Relevance: {summary['avg_relevance']}/10.0 (addresses query)",
                f"Completeness: {summary['avg_completeness']}/10.0 (covers all aspects)",
                f"Fluency: {summary['avg_fluency']}/10.0 (natural language)",
            ])
            
            if 'rating_distribution' in summary:
                report.extend([
                    "",
                "=== RATING DISTRIBUTION ===",
                f"Excellent (9.0-10.0): {summary['rating_distribution']['Excellent']} evaluations",
                f"Good (8.0-8.9): {summary['rating_distribution']['Good']} evaluations",
                f"Fair (6.5-7.9): {summary['rating_distribution']['Fair']} evaluations",
                f"Average (5.0-6.4): {summary['rating_distribution']['Average']} evaluations",
                f"Poor / Weak (<5.0): {summary['rating_distribution']['Poor / Weak']} evaluations",
                ])
            
            if 'judge_models' in summary:
                report.extend([
                    "",
                    "=== EVALUATION BY JUDGE MODEL ===",
                ])
                for judge, stats in summary['judge_models'].items():
                    report.append(f"{judge}: {stats['avg_score']}/10.0 ({stats['count']} evaluations)")
        
        report.extend([
            "",
            "RATING SCALE:",
            "9.0 – 10.0: Excellent",
            "8.0 – 8.9: Good", 
            "6.5 – 7.9: Fair",
            "5.0 – 6.4: Average",
            "< 5.0: Poor / Weak",
            "",
            f"Response Model: {self.current_session_metrics[0].model if self.current_session_metrics else 'N/A'}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "========================================="
        ])
        
        return "\n".join(report)

def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.
    """
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

def create_simple_vector_db(file_upload) -> Chroma:
    """Create a simple vector DB without complex embeddings to avoid the meta tensor error"""
    logger.info(f"Creating simple vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    try:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
        
        # Use PyPDFLoader for simplicity
        loader = PyPDFLoader(path)
        data = loader.load_and_split()
        
        # Simple text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Use a simpler embedding model that doesn't cause the meta tensor issue
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Smaller, more compatible model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}  # Simpler configuration
        )
        
        # Create vector store
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"pdf_{hash(file_upload.name)}"
        )
        
        logger.info("Simple vector DB created successfully")
        return vector_db
        
    except Exception as e:
        logger.error(f"Error creating vector DB: {e}")
        st.error(f"Error creating vector database: {str(e)}")
        raise
    finally:
        shutil.rmtree(temp_dir)

def get_simple_retriever(vector_db: Chroma):
    """Create a simple retriever without complex query expansion"""
    # Simple retriever configuration
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve 4 most similar documents
    )
    return retriever

def process_question_simple(question: str, vector_db: Chroma, selected_model: str) -> Tuple[str, str]:
    """Simple question processing without advanced features"""
    logger.info(f"Simple processing: {question}")
    
    llm = ChatOllama(model=selected_model, request_timeout=120.0)
    retriever = get_simple_retriever(vector_db)
    
    # Simple prompt template
    template = """You are a professional legal expert. 
    
    CONTEXT INFORMATION:
    {context}
    
    QUESTION: {question}
    
    Please provide a helpful answer based on the context above. If you cannot find the answer in the context, say so.
    
    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    # Get context for evaluation
    context_docs = retriever.invoke(question)
    context = "\n\n".join([
        f"Document {i+1}: {doc.page_content[:300]}..."
        for i, doc in enumerate(context_docs[:3])
    ])
    
    return response, context

def count_tokens(text: str) -> int:
    """Simple token counter"""
    return len(text.split())

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.title("🧠 DeepLaw - Legal RAG System")
    
    # Get available models
    try:
        models_info = ollama.list()
        available_models = extract_model_names(models_info)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        logger.error(f"Ollama connection error: {e}")
        available_models = tuple()
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "metrics_collector" not in st.session_state:
        st.session_state["metrics_collector"] = MetricsCollector()
    if "evaluation_enabled" not in st.session_state:
        st.session_state["evaluation_enabled"] = True

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection for response only
        if available_models:
            selected_model = st.selectbox(
                "Select Response Model", 
                available_models,
                key = "response_model_select",
                help = "Choose the model that will generate responses to your questions"
            )
        else:
            st.error("No Ollama models found. Please make sure Ollama is running.")
            st.stop()
        
        # Get judge models (all models except the selected one)
        judge_models = [model for model in available_models if model != selected_model]
        
        # Display judge models info
        st.header("👨‍⚖️ Judge Models")
        if judge_models:
            st.write(f"**{len(judge_models)} models** will evaluate each response:")
            for model in judge_models:
                st.write(f"• {model}")
        else:
            st.warning("No other models available for evaluation")
        
        # Evaluation toggle
        evaluation_enabled = st.toggle(
            "Enable Multi-Judge Evaluation", 
            value = True,
            key = "eval_toggle"
        )
        st.session_state["evaluation_enabled"] = evaluation_enabled
        
        # Initialize multi-judge evaluator
        if judge_models:
            judge_evaluator = LLMJudgeEvaluator(judge_models)
            st.session_state["judge_evaluator"] = judge_evaluator
        
        # Metrics actions
        st.header("📊 Evaluation Metrics")
        
        metrics_collector = st.session_state["metrics_collector"]
        
        if st.button("Show Evaluation Report"):
            report = metrics_collector.generate_report()
            st.text_area("Evaluation Report", report, height = 300)
        
        if st.button("Save All Metrics to File"):
            if metrics_collector.current_session_metrics:
                filename = metrics_collector.save_all_metrics_to_file()
                if filename:
                    st.success(f"All metrics saved to: {filename}")
                else:
                    st.error("Failed to save metrics.")
            else:
                st.warning("No metrics to save.")
        
        if st.button("Clear Metrics"):
            metrics_collector.current_session_metrics = []
            st.success("Metrics cleared.")
            
        # Saved metrics files section
        st.header("📁 Saved Evaluation Files")
        
        # Show list of evaluation files
        evaluation_files = []
        if os.path.exists(METRICS_DIR):
            evaluation_files = [f for f in os.listdir(METRICS_DIR) if f.endswith('.json') and f.startswith('evaluation_')]
            evaluation_files.sort(reverse = True)  # Newest first
        
        if evaluation_files:
            selected_file = st.selectbox("Select evaluation file", evaluation_files)
            
            if st.button("View Selected Evaluation"):
                filepath = os.path.join(METRICS_DIR, selected_file)
                try:
                    with open(filepath, 'r', encoding = 'utf-8') as f:
                        data = json.load(f)
                        st.json(data)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            
            if st.button("Download Selected Evaluation"):
                filepath = os.path.join(METRICS_DIR, selected_file)
                try:
                    with open(filepath, 'r', encoding = 'utf-8') as f:
                        data = f.read()
                        st.download_button(
                            label = "Download JSON",
                            data = data,
                            file_name = selected_file,
                            mime = "application/json"
                        )
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            st.info("No evaluation files found")

    # Main content
    # Regular file upload
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", 
        type = "pdf", 
        accept_multiple_files = False,
        key = "pdf_uploader"
    )

    if file_upload:
        if st.session_state["vector_db"] is None:
            with st.spinner("Processing uploaded PDF..."):
                try:
                    st.session_state["vector_db"] = create_simple_vector_db(file_upload)
                    st.session_state["file_upload"] = file_upload
                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    logger.error(f"PDF processing error: {e}")

    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value = 100, 
            max_value = 1000, 
            value = 700, 
            step = 50,
            key = "zoom_slider"
        )

        with col1:
            with st.container(height = 410, border = True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width = zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type = "secondary",
        key = "delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height = 500, border = True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar = avatar):
                st.markdown(message["content"])
                
                # Show evaluation scores if available
                if "evaluations" in message and message["evaluations"]:
                    # Handle both dictionary and object formats
                    if isinstance(message["evaluations"][0], dict):
                        avg_score = sum(eval_obj["overall_score"] for eval_obj in message["evaluations"]) / len(message["evaluations"])
                    else:
                        avg_score = sum(eval_obj.overall_score for eval_obj in message["evaluations"]) / len(message["evaluations"])
                    st.caption(f"📊 Average Evaluation: {avg_score:.1f}/10.0 ({len(message['evaluations'])} judges)")

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key = "chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar = "🤖"):
                    with st.spinner("Processing your question..."):
                        if st.session_state["vector_db"] is not None:
                            start_time = time.time()
                            response, context = process_question_simple(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            response_time = time.time() - start_time
                            token_count = count_tokens(response)
                            
                            st.markdown(response)
                            
                            # Record metrics if evaluation is enabled
                            if st.session_state["evaluation_enabled"] and "judge_evaluator" in st.session_state:
                                metrics_collector = st.session_state["metrics_collector"]
                                judge_evaluator = st.session_state["judge_evaluator"]
                                
                                # Get evaluations from all judge models
                                evaluations = judge_evaluator.evaluate_response(prompt, response, context)
                                
                                metrics = metrics_collector.record_metrics(
                                    query = prompt,
                                    response = response,
                                    context = context,
                                    response_time = response_time,
                                    token_count = token_count,
                                    model = selected_model,
                                    session_id = "streamlit_session",
                                    evaluations = evaluations
                                )
                                
                                # Convert evaluations to dictionaries for session state storage
                                eval_dicts = []
                                for eval_obj in evaluations:
                                    eval_dicts.append({
                                        "faithfulness": round(eval_obj.faithfulness, 1),
                                        "groundedness": round(eval_obj.groundedness, 1),
                                        "factual_consistency": round(eval_obj.factual_consistency, 1),
                                        "relevance": round(eval_obj.relevance, 1),
                                        "completeness": round(eval_obj.completeness, 1),
                                        "fluency": round(eval_obj.fluency, 1),
                                        "overall_score": round(eval_obj.overall_score, 1),
                                        "rating": eval_obj.evaluation_notes.split(":")[0],
                                        "judge_model": eval_obj.judge_model
                                    })
                                
                                st.session_state["messages"].append({
                                    "role": "assistant", 
                                    "content": response,
                                    "evaluations": eval_dicts
                                })
                            else:
                                st.session_state["messages"].append({
                                    "role": "assistant", 
                                    "content": response
                                })
                        else:
                            st.warning("Please upload a PDF file first.")

            except Exception as e:
                st.error(f"Error: {str(e)}", icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")

    # Display current metrics summary in sidebar
    metrics_collector = st.session_state["metrics_collector"]
    if metrics_collector.current_session_metrics:
        with st.sidebar:
            st.subheader("📈 Current Session Summary")
            summary = metrics_collector.get_session_summary()
            
            if summary:
                st.metric("Total Interactions", summary["total_interactions"])
                st.metric("Total Evaluations", summary["total_evaluations"])
                st.metric("Avg Response Time", f"{summary['avg_response_time']}s")
                
                if "avg_overall_score" in summary:
                    st.metric("Overall Quality", f"{summary['avg_overall_score']}/10.0")
                    
                    # Display rating distribution as a bar chart
                    if "rating_distribution" in summary:
                        st.subheader("Rating Distribution")
                        rating_data = summary["rating_distribution"]
                        st.bar_chart(rating_data)

if __name__ == "__main__":
    main()