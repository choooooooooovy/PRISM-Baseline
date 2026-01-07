import os
import logging
from openai import OpenAI
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    def generate_options(
        self,
        step0_data: Dict[str, Any],
        step1_data: Dict[str, Any],
        step2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate career/decision options based on Steps 0-2 data using LLM
        
        Args:
            step0_data: Self-profile data (values, interests, strengths, constraints, concerns)
            step1_data: Communication data (problem definition, cues, questions)
            step2_data: Analysis data (evaluation criteria, constraints, information template)
        
        Returns:
            Dict containing generated options with titles, descriptions, and profiles
        """
        
        # Build structured prompt
        prompt = self._build_prompt(step0_data, step1_data, step2_data)
        
        try:
            logger.info(f"Sending request to OpenAI ({self.model})")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a career counseling expert specializing in the CASVE (Communication, Analysis, Synthesis, Valuing, Execution) decision-making model. 
Your task is to generate personalized career or decision alternatives based on the user's profile, problem definition, and analysis criteria.

Generate 3-5 realistic and actionable options that:
1. Align with the user's values, interests, and strengths
2. Consider their constraints and concerns
3. Address their decision-making problem
4. Can be evaluated using their criteria

Return the response in JSON format with the following structure:
{
  "options": [
    {
      "title": "Option title (brief, clear)",
      "description": "2-3 sentence overview",
      "profile": {
        "coreRole": "Main role/responsibility",
        "requiredSkills": "Key skills needed",
        "environment": "Work environment description",
        "growth": "Growth potential and trajectory"
      },
      "matchReason": "Why this fits the user (2-3 sentences)"
    }
  ]
}

Respond ONLY with valid JSON, no additional text."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            tokens_used = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"OpenAI response received. Tokens used: {tokens_used['total_tokens']}")
            
            return {
                "success": True,
                "content": content,
                "tokens_used": tokens_used,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_prompt(
        self,
        step0_data: Dict[str, Any],
        step1_data: Dict[str, Any],
        step2_data: Dict[str, Any]
    ) -> str:
        """Build structured prompt from Steps 0-2 data"""
        
        prompt_parts = ["# User Profile and Decision Context\n"]
        
        # Step 0: Self Profile
        prompt_parts.append("## Step 0: Self Profile")
        prompt_parts.append(f"**Values**: {', '.join(step0_data.get('values', []))}")
        prompt_parts.append(f"**Interests**: {', '.join(step0_data.get('interests', []))}")
        prompt_parts.append(f"**Strengths**: {', '.join(step0_data.get('strengths', []))}")
        
        must_have = step0_data.get('mustHaveConstraints', [])
        nice_to_have = step0_data.get('niceToHaveConstraints', [])
        if must_have:
            prompt_parts.append(f"**Must-Have Constraints**: {', '.join(must_have)}")
        if nice_to_have:
            prompt_parts.append(f"**Nice-to-Have Constraints**: {', '.join(nice_to_have)}")
        
        concerns = step0_data.get('concerns', '')
        if concerns:
            prompt_parts.append(f"**Current Concerns**: {concerns}")
        
        # Step 1: Communication
        prompt_parts.append("\n## Step 1: Problem Definition")
        problem_def = step1_data.get('problemDefinition', '')
        if problem_def:
            prompt_parts.append(f"**Decision Problem**: {problem_def}")
        
        internal_cues = step1_data.get('internalCues', [])
        external_cues = step1_data.get('externalCues', [])
        if internal_cues:
            prompt_parts.append(f"**Internal Signals**: {', '.join(internal_cues)}")
        if external_cues:
            prompt_parts.append(f"**External Signals**: {', '.join(external_cues)}")
        
        key_questions = step1_data.get('keyQuestions', [])
        if key_questions:
            prompt_parts.append(f"**Key Questions**: {', '.join(key_questions)}")
        
        # Step 2: Analysis
        prompt_parts.append("\n## Step 2: Evaluation Criteria")
        criteria = step2_data.get('evaluationCriteria', [])
        if criteria:
            prompt_parts.append(f"**Comparison Criteria**: {', '.join(criteria)}")
        
        constraints = step2_data.get('constraints', [])
        if constraints:
            prompt_parts.append(f"**Additional Constraints**: {', '.join(constraints)}")
        
        prompt_parts.append("\n---")
        prompt_parts.append("Based on this information, generate 3-5 personalized career/decision options.")
        
        return "\n".join(prompt_parts)
