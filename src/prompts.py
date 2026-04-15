def get_train_prompt_v5(input_text: str, original_text: str) -> str:
    INSTRUCTION_v5 = "Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück."
    return f"<s>[INST]{INSTRUCTION_v5}\n\n{input_text}[/INST]{original_text}</s>"


def get_inference_prompt_v5(input_text: str) -> str:
    INSTRUCTION_v5 = "Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück."
    return f"<s>[INST]{INSTRUCTION_v5}\n\n{input_text}[/INST]"


def get_inference_prompt_v6(input_text: str) -> str:
    return f"<s>[INST]{instruction_v6.format(input_text=input_text)}[/INST]"


instruction_v6 = """Du bist ein strenger und hochpräziser Korrektor für deutsche Grammatik. Deine einzige Aufgabe ist es, Grammatik-, Rechtschreib- und Zeichensetzungsfehler im bereitgestellten Text zu beheben und das Ergebnis als valides JSON zurückzugeben.

STRIKTE REGELN:
1. JSON-FORMAT: Die Antwort muss ein gültiges JSON-Objekt sein, das exakt einen Key enthält: "corrected_text".
2. MINIMALE ÄNDERUNGEN: Verändere niemals den ursprünglichen Stil, Ton oder das Vokabular. Mach den Text nicht formeller, als er ist. Nutze keine **Markdown-Formatierungen** und füge keine zusätzlichen Erklärungen hinzu.
3. KEIN CHAT & KEINE CODEBLÖCKE: Gib AUSSCHLIESSLICH das JSON-Objekt zurück. Füge keinen Text vor oder nach dem JSON hinzu. Verwende KEINE Markdown-Formatierungen (wie ```json ... ```).
4. KEINE ÜBERKORREKTUR: Wenn der Satz korrekt ist, gib ihn unverändert im JSON zurück.

BEISPIELE:

Input: "ich gehe heute in den stadt weil ich einkaufen muss"
Output: {{"corrected_text": "Ich gehe heute in die Stadt, weil ich einkaufen muss."}}

Input: "Das is mir echt voll egal was die anderen sagen."
Output: {{"corrected_text": "Das ist mir echt voll egal, was die anderen sagen."}}

Input: "Wir standen nur da und schauten einander überrascht an."
Output: {{"corrected_text": "Wir standen nur da und schauten einander überrascht an."}}

TEXT ZUR KORREKTUR:
{input_text}
Output:
""" 


if __name__ == "__main__":
    # t = instruction_v6.format(input_text="Das is mir echt voll egal was die anderen sagen.")
    t = get_inference_prompt_v6("Das is mir echt voll egal was die anderen sagen.")
    print(t)