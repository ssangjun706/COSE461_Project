import pandas as pd
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Any


def build_meta_prompt(X: Any, target: str, target_values: list[str]) -> str:
    return f"""You are an expert prompt engineer who creates sophisticated natural language prompts for LLMs (Predictors), based on structured input data.

    Your task is to convert the given input data into an engaging, contextually rich natural language prompt that enables a language model to accurately predict the value of the target column '{target}'.

    The generated prompt must follow these guidelines:

    1. **Must** use the input data values as the foundation, but feel free to rephrase or contextualize them naturally.
    2. Select the most relevant fields from the input that would help with prediction accuracy.
    3. Leverage common knowledge to create contextually rich prompts that frame the prediction task in a meaningful way.
    4. Direct the language model to respond with one of these specific values: {target_values}.
    5. Craft varied, sophisticated, and natural language using diverse sentence structures, appropriate domain terminology, and engaging phrasing.
    6. Ensure the prompt flows naturally while clearly communicating all relevant input information.
    7. Consider the semantic relationships between fields to create a coherent narrative.
    8. **Critical**: Your prompt MUST explicitly instruct the model to respond with ONLY the prediction value, with no explanations, no step-by-step reasoning, and no additional text.

    Below is the input data in a structured format:

    {X}

    Now, generate a high-quality natural language prompt for predicting '{target}'.
    The model should respond with one of: {target_values}.

    **IMPORTANT: The final part of your prompt must contain these exact instructions:**
    "Based on the information provided, respond with ONLY {target_values}. Do not include any explanations, reasoning, or additional text. Do not use phrases like 'I think' or 'The answer is'. Your entire response must be only the prediction value."
    """

#     strategies = [
#         {
#             "name": "Domain Expert Reasoning",
#             "description": "Think like a domain expert who understands the underlying factors and relationships",
#             "guidance": "Consider what a subject matter expert would know about this type of prediction task. What hidden patterns, causal relationships, or contextual factors might be relevant?",
#             "example_phrases": ["experts in this field recognize", "domain knowledge suggests", "experienced practitioners know"]
#         },
#         {
#             "name": "Historical Pattern Analysis", 
#             "description": "Analyze the data through the lens of historical context and established patterns",
#             "guidance": "Think about the historical, social, or scientific context surrounding this data. What broader patterns or established principles might apply?",
#             "example_phrases": ["historical evidence shows", "established patterns indicate", "contextual analysis reveals"]
#         },
#         {
#             "name": "Causal Factor Investigation",
#             "description": "Investigate the underlying causal relationships between features and outcomes",
#             "guidance": "Look beyond surface-level correlations to understand why certain factors might influence the outcome. What are the underlying mechanisms?",
#             "example_phrases": ["causal analysis suggests", "underlying mechanisms indicate", "root cause examination shows"]
#         },
#         {
#             "name": "Comparative Demographic Analysis",
#             "description": "Compare different demographic groups and analyze differential outcomes",
#             "guidance": "Consider how different groups might experience different outcomes and why. What systematic differences might exist?",
#             "example_phrases": ["demographic analysis reveals", "group comparisons show", "population studies indicate"]
#         },
#         {
#             "name": "Risk-Benefit Assessment",
#             "description": "Systematically evaluate risk and protective factors",
#             "guidance": "Think about what factors might increase or decrease the likelihood of the outcome. Consider both obvious and subtle influences.",
#             "example_phrases": ["risk assessment indicates", "protective factors include", "vulnerability analysis shows"]
#         },
#         {
#             "name": "Systems Thinking Approach",
#             "description": "Consider the broader system and environmental factors that might influence outcomes",
#             "guidance": "Think about the larger context, environment, or system in which this data exists. What external factors might be at play?",
#             "example_phrases": ["systems analysis suggests", "environmental factors indicate", "contextual influences show"]
#         }
#     ]

#     selected_strategy = random.choice(strategies)
    
#     discovery_prompts = [
#         "What domain-specific knowledge would be most relevant for this type of prediction?",
#         "What contextual factors or background information should be considered?",
#         "What would a subject matter expert immediately recognize about this data?",
#         "What historical, social, or scientific principles might apply here?",
#         "What underlying patterns or relationships might not be immediately obvious?",
#         "What external knowledge would help interpret these data points more accurately?"
#     ]
    
#     selected_discovery = random.sample(discovery_prompts, 2)
    
#     return f"""You are an expert prompt engineer who creates highly effective prediction prompts. Your goal is to transform structured data into a compelling natural language prompt that leverages relevant domain knowledge and maximizes prediction accuracy.

# **STRATEGIC APPROACH: {selected_strategy['name']}**
# {selected_strategy['description']}

# **GUIDANCE FOR THIS APPROACH:**
# {selected_strategy['guidance']}

# **DOMAIN KNOWLEDGE DISCOVERY:**
# Before creating your prompt, consider these questions:
# - {selected_discovery[0]}
# - {selected_discovery[1]}

# **INPUT DATA:**
# {X}

# **TARGET PREDICTION:** {target} (possible values: {target_values})

# **PROMPT CREATION INSTRUCTIONS:**

# 1. **Domain Knowledge Integration**: 
#    - Draw upon your understanding of relevant domain knowledge
#    - Consider what contextual information would be important for this type of prediction
#    - Think about underlying factors, historical patterns, or established principles that might apply
#    - Use your knowledge to interpret the significance of each data point

# 2. **Strategic Reasoning**: 
#    - Apply the {selected_strategy['name']} method
#    - Use language patterns like: {', '.join(selected_strategy['example_phrases'])}
#    - Structure your reasoning according to this approach

# 3. **Data Interpretation**:
#    - Don't just list the data points - explain their significance
#    - Consider interactions between different variables
#    - Highlight the most predictive factors based on domain understanding
#    - Address potential confounding factors or nuances

# 4. **Contextual Analysis**:
#    - Place the data in its broader context
#    - Consider what external factors might influence the outcome
#    - Think about systematic patterns or group differences
#    - Apply relevant background knowledge to enhance interpretation

# 5. **Prediction Guidance**:
#    - Structure the prompt to guide toward the most relevant signals
#    - Balance multiple factors appropriately
#    - Lead directly to a prediction without encouraging reasoning steps

# **QUALITY STANDARDS:**
# - Demonstrate deep understanding through specific insights
# - Use concrete, evidence-based reasoning
# - Maintain logical flow from context → analysis → prediction demand
# - Include quantitative reasoning when relevant
# - Ensure the prompt is comprehensive yet focused
# - ABSOLUTELY PREVENT any step-by-step thinking outputs

# **OUTPUT REQUIREMENTS:**
# Create a prediction prompt that:
# 1. Establishes relevant context using domain knowledge
# 2. Analyzes the provided data through an expert lens
# 3. Applies the selected strategic approach consistently
# 4. Guides toward an informed prediction with clear, direct instructions
# 6. Concludes with a strong directive: "**IMPORTANT:** Respond with ONLY {target_values}. Do not include any explanations, reasoning, or additional text."

# **Remember:** Your prompt should demonstrate expert-level thinking while absolutely preventing the target model from showing any reasoning process. The goal is immediate, direct prediction without any thinking steps visible in the output.

# Generate your expert-level prediction prompt now:"""


class BaseDataset(Dataset):
    """Base interface for datasets used in the fine-tuning pipeline"""

    def __init__(
        self,
        path: str,
        target: str,
        target_values: list[str],
        train: bool = True,
        shuffle: bool = True,
        train_size: float = 0.8,
    ):
        super().__init__()
        self.target = target
        self.target_values = target_values
        self.feature_labels = []

        self.X = None
        self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        raise NotImplementedError("Subclasses must implement this method")


class TitanicDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        train: bool = True,
        shuffle: bool = True,
        train_size: float = 0.8,
    ):
        super().__init__(
            path=path,
            target="Survived",
            target_values=["DEAD", "ALIVE"],
            train=train,
            shuffle=shuffle,
            train_size=train_size,
        )

        df = pd.read_csv(path)

        assert (
            self.target in df.columns
        ), f"Target column '{self.target}' not found in the dataframe."

        self.feature_labels = df.drop(columns=[self.target]).columns.tolist()

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=42,
            shuffle=shuffle,
            stratify=y,
        )

        self.X = X_train if train else X_test
        self.y = y_train if train else y_test

    def __getitem__(self, idx: int) -> tuple[str, str]:
        X = build_meta_prompt(
            self.X.iloc[idx].to_dict(), self.target, self.target_values
        )
        y = self.target_values[self.y.iloc[idx].item()]
        return X, y
