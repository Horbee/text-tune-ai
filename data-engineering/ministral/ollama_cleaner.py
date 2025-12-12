import pandas as pd
import ollama
import json
import os

df = pd.read_json('ministral-data.jsonl', lines=True)

output_file = 'ministral-corrected.jsonl'

# Check for existing progress to resume
start_idx = 0
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        start_idx = sum(1 for _ in f)
    if start_idx > 0:
        print(f"Resuming from row {start_idx} (found {start_idx} existing results)")

for idx in range(start_idx, len(df)):
    print(f"\nProcessing row: {idx}...")

    text_content = df.iloc[idx].get('text')

    try:
        response = ollama.chat(model='ministral-3:8b', messages=[
            {
            "role": "system",
            "content": '''Du bist ein Experte für deutsche Grammatik und Rechtschreibung. Du erhältst einen Satz, der möglicherweise Fehler enthält.

                Deine Aufgabe:
                1. Korrigiere alle grammatikalischen Fehler (Kasus, Tempus, Konjugation, Deklination).
                2. Korrigiere die Satzstruktur (korrekte Wortstellung im Haupt- und Nebensatz).
                3. Korrigiere die Zeichensetzung (Kommas, Punkte, Anführungszeichen).
                4. Entferne unerwünschte Annotationen, Klammern oder Metadaten.
                5. Stelle sicher, dass der Satz mit einem Großbuchstaben beginnt und mit einem korrekten Satzzeichen endet (. ! ?).
                6. Falls der Satz bereits korrekt ist, gib ihn unverändert zurück.

                WICHTIG: Antworte NUR mit dem korrigierten Satz. Keine Erklärungen, keine Kommentare, keine Anführungszeichen um den Satz.

                Beispiele:
                Eingabe: "Die Polizei habe dem Ermittlungen aufgenommen."
                Ausgabe: Die Polizei hat die Ermittlungen aufgenommen.

                Eingabe: "weil er gestern nach Hause gegangen ist war er müde"
                Ausgabe: Weil er gestern nach Hause gegangen ist, war er müde.

                Eingabe: "Der Mann, der das Buch liest"
                Ausgabe: Der Mann, der das Buch liest.'''
            },
            {
                'role': 'user',
                'content': text_content,
            }
        ])
        
        result = response['message']['content'].strip()
        print(f"Model response:\n{result}\n")
        
        # Append single result to file (much more efficient than rewriting entire file)
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump({'corrected_text': result, 'original_text': text_content}, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Progress saved: {idx + 1}/{len(df)} rows")
                    
    except Exception as e:
        print(f"Error processing row {idx}: {e}")


print(f"Completed! Results written to '{output_file}'")