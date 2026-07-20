"""GPU Workload Prediction Agent (GWPA) - Prototype.

This agent uses an LLM to reason about GPU workload patterns and make predictions.
It combines statistical tools with natural language reasoning.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

class StatisticalTools:
    """Tools for the agent to analyze time series data."""
    
    @staticmethod
    def get_basic_stats(series):
        return {
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
        }
    
    @staticmethod
    def get_trend(series):
        """Simple linear regression to determine trend."""
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        return {"direction": direction, "slope": float(slope)}
    
    @staticmethod
    def check_periodicity(series, period=144):
        """Check if there's a pattern at a specific period (e.g., daily)."""
        if len(series) < period * 2:
            return {"has_pattern": False}
        
        # Simple autocorrelation check
        corr = np.corrcoef(series[:-period], series[period:])[0, 1]
        return {"has_pattern": bool(corr > 0.5), "correlation": float(corr)}


class GWPAgent:
    def __init__(self):
        print(f"[agent] loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tools = StatisticalTools()
        print("[agent] ready.")

    def observe(self, context_window):
        """Step 1: Perception - Analyze the context window."""
        stats_info = self.tools.get_basic_stats(context_window)
        trend_info = self.tools.get_trend(context_window)
        periodicity_info = self.tools.check_periodicity(context_window)
        
        observation = (
            f"Observation:\n"
            f"- Mean utilization: {stats_info['mean']:.2f}%\n"
            f"- Std deviation: {stats_info['std']:.2f}%\n"
            f"- Range: [{stats_info['min']:.2f}%, {stats_info['max']:.2f}%]\n"
            f"- Trend: {trend_info['direction']} (slope: {trend_info['slope']:.4f})\n"
            f"- Daily pattern (144 steps): {'Detected' if periodicity_info['has_pattern'] else 'Not detected'} "
            f"(corr: {periodicity_info.get('correlation', 0):.2f})\n"
        )
        return observation

    def reason_and_predict(self, context_window, pred_len=144):
        """Step 2 & 3: Reasoning and Prediction."""
        observation = self.observe(context_window)
        
        # Calculate a naive baseline for the agent to reference
        last_val = context_window[-1]
        mean_val = np.mean(context_window)
        
        prompt = f"""You are an expert AI Analyst for GPU Clusters. Your goal is to predict future workload based on historical patterns.

### Data Analysis Report
{observation}

### Context
- The last observed value was {last_val:.2f}%.
- The historical average is {mean_val:.2f}%.
- We need to predict the next {pred_len} time steps (each step is 10 minutes).

### Reasoning Instructions
1. **Analyze the Trend**: Is the workload increasing, decreasing, or stable?
2. **Identify Patterns**: Do you see signs of daily cycles (periodicity)?
3. **Assess Volatility**: Is the standard deviation high (unstable) or low (predictable)?
4. **Formulate Strategy**: Based on the above, should the prediction follow the recent trend, revert to the mean, or expect a spike?

### Output Format
Please provide your analysis in the following format:
**Analysis**: [Your step-by-step reasoning here]
**Strategy**: [e.g., "Follow upward trend", "Revert to mean due to high volatility"]
**Prediction Summary**: [A brief description of the expected curve, e.g., "Gradual increase from 50% to 65%"]

Let's think step by step:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        print("[agent] generating professional analysis...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt_text):]
        
        print("\n" + "-"*50)
        print("AGENT ANALYSIS REPORT:")
        print("-"*50)
        print(response)
        print("-"*50 + "\n")
        return response


def main():
    # Load some sample data
    data_path = "/home/hongshao.hzx/opendc-bench/data/alibaba_10min_test_v2.npy"
    try:
        series = np.load(data_path).astype(np.float32)
    except FileNotFoundError:
        print("[error] test data not found, using random data for demo")
        series = np.random.rand(500).astype(np.float32) * 100

    agent = GWPAgent()
    
    # Run the agent on a window
    ctx_len = 288
    context = series[:ctx_len]
    
    print("\n" + "="*50)
    print("Starting GPU Workload Prediction Agent Demo")
    print("="*50)
    
    reasoning = agent.reason_and_predict(context)
    
    print("\n" + "="*50)
    print("Agent Analysis Complete")
    print("="*50)


if __name__ == "__main__":
    main()
