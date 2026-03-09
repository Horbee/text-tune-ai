def get_train_prompt_v5(input_text: str, original_text: str) -> str:
    INSTRUCTION_v5 = "Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück."
    return f"<s>[INST]{INSTRUCTION_v5}\n\n{input_text}[/INST]{original_text}</s>"


def get_inference_prompt_v5(input_text: str) -> str:
    INSTRUCTION_v5 = "Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück."
    return f"<s>[INST]{INSTRUCTION_v5}\n\n{input_text}[/INST]"