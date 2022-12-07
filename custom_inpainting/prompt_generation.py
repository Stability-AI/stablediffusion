import random
from pathlib import Path


class PromptGenerator:

    def __init__(
            self, 
            base_prompt: str, 
            tags_txt_path: str = None, 
            number_of_tags_per_prompt: int = 1
        ) -> None:
        if tags_txt_path is not None:   
            self.tags  = [tag.strip() for tag in Path(tags_txt_path).read_text().splitlines()]
        else:
            self.tags = None
        self.base_prompt = base_prompt
        self.number_of_tags_per_prompt = number_of_tags_per_prompt


    def generate_prompt(self) -> str:
        prompt_parts = [self.base_prompt]
        if self.tags is not None:
            random.shuffle(self.tags)
            prompt_parts.extend(self.tags[:self.number_of_tags_per_prompt])
        return ", ".join(prompt_parts)
