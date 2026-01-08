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
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
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
        
        # Build dynamic system prompt based on informationTemplate
        info_template = step2_data.get('informationTemplate', [])
        system_prompt = self._build_system_prompt(info_template)
        
        try:
            logger.info(f"Sending request to OpenAI ({self.model})")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
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
        prompt_parts.append("Based on this information, generate exactly 5 personalized career/decision options.")
        
        return "\n".join(prompt_parts)
    
    def _build_system_prompt(self, info_template: list) -> str:
        """Build dynamic system prompt based on informationTemplate"""
        
        # Build profile field list from informationTemplate
        profile_fields = []
        for item in info_template:
            field_name = item.get('field', '')
            description = item.get('description', '')
            # Extract field key from "한글이름 (fieldKey)" format
            if '(' in field_name and ')' in field_name:
                field_key = field_name.split('(')[1].split(')')[0]
                field_label = field_name.split('(')[0].strip()
            else:
                field_key = field_name.replace(' ', '_').lower()
                field_label = field_name
            
            profile_fields.append({
                'key': field_key,
                'label': field_label,
                'description': description
            })
        
        # Build profile JSON structure
        profile_json_fields = []
        for field in profile_fields:
            profile_json_fields.append(f'        "{field["key"]}": "{field["label"]} - {field["description"]}"')
        
        profile_json_str = ',\n'.join(profile_json_fields)
        
        system_prompt = f"""당신은 CASVE (Communication, Analysis, Synthesis, Valuing, Execution) 의사결정 모델을 전문으로 하는 진로 상담 전문가입니다.
사용자의 프로필, 문제 정의, 분석 기준을 바탕으로 개인화된 진로 또는 의사결정 대안을 생성하는 것이 당신의 임무입니다.

다음 조건을 충족하는 현실적이고 실행 가능한 대안을 정확히 5개 생성하세요:
1. 사용자의 가치관, 흥미, 강점과 일치
2. 제약 조건과 우려 사항 고려
3. 의사결정 문제 해결
4. 평가 기준으로 비교 가능

응답은 다음 JSON 형식으로만 작성하세요:
{{
  "options": [
    {{
      "title": "대안 제목 (간결하고 명확하게)",
      "description": "2-3문장 개요",
      "profile": {{
{profile_json_str}
      }},
      "matchReason": "사용자에게 적합한 이유 (2-3문장)"
    }}
  ]
}}

반드시 유효한 JSON으로만 응답하고, 추가 텍스트는 작성하지 마세요. 모든 응답은 한글로 작성하세요.
각 profile 필드는 위에 명시된 키를 정확히 사용하고, 해당 필드에 대한 구체적이고 유용한 정보를 제공하세요."""
        
        return system_prompt
