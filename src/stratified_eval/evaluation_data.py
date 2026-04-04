"""
Tier 1: Localized Mechanics (Easy)
Simple spelling mistakes, wrong capitalization, or localized subject-verb agreement errors.
"""

TIER1_EXAMPLES = [
    {
        "corrupted": "Das Kind spielen im Garten.",
        "original": "Das Kind spielt im Garten.",
        "error_types": ["subject-verb_agreement"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Ich geht morgen einkaufen.",
        "original": "Ich gehe morgen einkaufen.",
        "error_types": ["verb_conjugation"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Der hund ist sehr groß.",
        "original": "Der Hund ist sehr groß.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Sie habe einen hund.",
        "original": "Sie hat einen Hund.",
        "error_types": ["verb_conjugation", "capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Wir spielen Fussball morgen.",
        "original": "Wir spielen Fußball morgen.",
        "error_types": ["orthography"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Das ist eine gute idee.",
        "original": "Das ist eine gute Idee.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Ich kann nicht kommen weil ich krank bin.",
        "original": "Ich kann nicht kommen, weil ich krank bin.",
        "error_types": ["punctuation"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Er ist ein gut mann.",
        "original": "Er ist ein guter Mann.",
        "error_types": ["adjective_ending"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Die katze schlaft auf dem sofa.",
        "original": "Die Katze schläft auf dem Sofa.",
        "error_types": ["capitalization", "orthography"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Wir haben viele schöne erinnerungen.",
        "original": "Wir haben viele schöne Erinnerungen.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Er lesst das Buch.",
        "original": "Er liest das Buch.",
        "error_types": ["orthography"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Ich habe hunger.",
        "original": "Ich habe Hunger.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Das kind spielt mit der puppe.",
        "original": "Das Kind spielt mit der Puppe.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Sie gehen zur schule.",
        "original": "Sie gehen zur Schule.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
    {
        "corrupted": "Ich mag die blumen.",
        "original": "Ich mag die Blumen.",
        "error_types": ["capitalization"],
        "difficulty": "easy",
    },
]

"""
Tier 2: Context-Dependent Grammar (Medium)
Case and gender mistakes that require understanding the whole sentence structure.
"""
TIER2_EXAMPLES = [
    {
        "corrupted": "Ich gebe dem Mann das Buch.",
        "original": "Ich gebe dem Mann das Buch.",
        "error_types": ["case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Er hat ein Problem mit sein Vater.",
        "original": "Er hat ein Problem mit seinem Vater.",
        "error_types": ["case", "possessive_pronoun"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Die Frau von den Mann ist nett.",
        "original": "Die Frau von dem Mann ist nett.",
        "error_types": ["preposition_case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Ich erinnere mich an die alte Lehrerin die uns geholfen hat.",
        "original": "Ich erinnere mich an die alte Lehrerin, die uns geholfen hat.",
        "error_types": ["relative_clause_comma"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Der Hund belongt dem Madchen.",
        "original": "Der Hund gehört dem Mädchen.",
        "error_types": ["orthography", "case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Sie spricht mit ihr Schwester.",
        "original": "Sie spricht mit ihrer Schwester.",
        "error_types": ["possessive_pronoun_case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Ich habe den Film schon gesehen.",
        "original": "Ich habe den Film schon gesehen.",
        "error_types": ["article_case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Nach dem Essen, gehen wir spazieren.",
        "original": "Nach dem Essen gehen wir spazieren.",
        "error_types": ["punctuation"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Das ist die Mann dessen Auto kaputt ist.",
        "original": "Das ist der Mann, dessen Auto kaputt ist.",
        "error_types": ["relative_pronoun_case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Er hat mir ein Geschenk gegeben dass mich sehr gefreut hat.",
        "original": "Er hat mir ein Geschenk gegeben, das mich sehr gefreut hat.",
        "error_types": ["relative_clause_comma"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Ich freue mich auf die Reise nach Italia.",
        "original": "Ich freue mich auf die Reise nach Italien.",
        "error_types": ["orthography"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Der Konig von England ist beliebt.",
        "original": "Der König von England ist beliebt.",
        "error_types": ["orthography"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Sie hat mit ihr Freund gesprochen.",
        "original": "Sie hat mit ihrem Freund gesprochen.",
        "error_types": ["possessive_pronoun_case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Ich traue dem mann nicht.",
        "original": "Ich traue dem Mann nicht.",
        "error_types": ["capitalization", "case"],
        "difficulty": "medium",
    },
    {
        "corrupted": "Das gehort zu mir.",
        "original": "Das gehört zu mir.",
        "error_types": ["orthography"],
        "difficulty": "medium",
    },
]

"""
Tier 3: Complex Restructuring (Hard)
Correcting broken passive voice, deeply nested subordinate clauses, or heavy tense inconsistencies.
"""
TIER3_EXAMPLES = [
    {
        "corrupted": "Gestern habe ich see you tomorrow gesagt.",
        "original": "Gestern habe ich gesagt, dass ich dich morgen sehe.",
        "error_types": ["word_order", "tense_mixing", "language_mixing"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Wenn ich Zeit gehabt habe, ich hatte das Buch gelesen.",
        "original": "Wenn ich Zeit gehabt hätte, hätte ich das Buch gelesen.",
        "error_types": ["subjunctive", "word_order"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Er wurde gesagt dass das Kind krank ist.",
        "original": "Ihm wurde gesagt, dass das Kind krank ist.",
        "error_types": ["passive_voice", "word_order"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Ich wunschte dass ich dahaime fleissig gearbeitet habe.",
        "original": "Ich wünschte, dass ich dauerhaft fleißig gearbeitet hätte.",
        "error_types": ["orthography", "subjunctive", "tense"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Obwohl er war mude, hat er weitergearbeitet.",
        "original": "Obwohl er müde war, hat er weitergearbeitet.",
        "error_types": ["word_order", "adjective_decline"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Der Mann der gestern kam und das Essen brachte ist mein Vater.",
        "original": "Der Mann, der gestern kam und das Essen brachte, ist mein Vater.",
        "error_types": ["relative_clause_comma", "word_order"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Ich habe gelernt dass wenn man fleissig ist dass man dann Erfolg hat.",
        "original": "Ich habe gelernt, dass man Erfolg hat, wenn man fleißig ist.",
        "error_types": ["nested_clauses", "word_order"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Der hund ist geloffen worden gestern im park.",
        "original": "Der Hund ist gestern im Park gelaufen worden.",
        "error_types": ["passive_voice", "word_order", "capitalization"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Weil er hat nicht kommen konnen weil er war krank.",
        "original": "Weil er nicht hat kommen können, weil er krank war.",
        "error_types": ["word_order", "infinitive_construction"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Das ist die katze von dem nachbar die immer auf dem dach schlaft.",
        "original": "Das ist die Katze von dem Nachbarn, die immer auf dem Dach schläft.",
        "error_types": ["capitalization", "relative_clause"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Er hat gesagt dass er wird kommen hatte morgen.",
        "original": "Er hat gesagt, dass er morgen kommen würde.",
        "error_types": ["word_order", "tense", "subjunctive"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Die Kinder haben been ge-schpeelt den ganze tag.",
        "original": "Die Kinder haben den ganzen Tag gespielt.",
        "error_types": ["language_mixing", "orthography", "word_order"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Wenn ich reich bin wurde ich ein haus kaufen.",
        "original": "Wenn ich reich wäre, würde ich ein Haus kaufen.",
        "error_types": ["subjunctive", "word_order"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Ich habe nichts tun konnen weil ich muss arbeiten.",
        "original": "Ich habe nichts tun können, weil ich arbeiten musste.",
        "error_types": ["infinitive_construction", "tense"],
        "difficulty": "hard",
    },
    {
        "corrupted": "Der professor erklart dass die studenten gelernt haben mussen fur die prufung.",
        "original": "Der Professor erklärt, dass die Studenten für die Prüfung lernen mussten.",
        "error_types": ["word_order", "tense", "orthography"],
        "difficulty": "hard",
    },
]

"""
Flawless sentences for Over-Correction Benchmark (FPR testing)
"""
FLAWLESS_EXAMPLES = [
    {"original": "Der Hund läuft im Garten.", "source": "standard"},
    {"original": "Ich habe gestern einen Film gesehen.", "source": "standard"},
    {"original": "Die Kinder spielen in der Schule.", "source": "standard"},
    {"original": "Er ist sehr müde heute.", "source": "standard"},
    {"original": "Wir fahren morgen nach Berlin.", "source": "standard"},
    {"original": "Das Buch ist interessant.", "source": "standard"},
    {"original": "Sie trinkt gerne Kaffee.", "source": "standard"},
    {"original": "Ich komme aus Deutschland.", "source": "standard"},
    {"original": "Er hat zwei Kinder.", "source": "standard"},
    {"original": "Die Sonne scheint heute.", "source": "standard"},
    {"original": "Wir essen zum Frühstück Brot.", "source": "standard"},
    {"original": "Das Auto ist rot.", "source": "standard"},
    {"original": "Er arbeitet in einem Büro.", "source": "standard"},
    {"original": "Ich lerne Deutsch.", "source": "standard"},
    {"original": "Sie hat schöne Haare.", "source": "standard"},
    {"original": "Das Essen schmeckt gut.", "source": "standard"},
    {"original": "Wir wohnen in München.", "source": "standard"},
    {"original": "Er liest gerne Bücher.", "source": "standard"},
    {"original": "Die Katze schläft auf dem Sofa.", "source": "standard"},
    {"original": "Ich habe einen Hund.", "source": "standard"},
]

"""
Stress test examples with various edge cases
"""
STRESS_TEST_LENGTH = [
    {"text": "Der Hund.", "word_count": 2, "category": "very_short"},
    {
        "text": "Der Hund läuft im Garten und spielt mit dem Ball.",
        "word_count": 10,
        "category": "short",
    },
    {
        "text": "Der Hund läuft im Garten und spielt mit dem Ball während die Kinder zuschauen und lachen.",
        "word_count": 16,
        "category": "medium",
    },
    {
        "text": "Der Hund läuft im Garten und spielt mit dem Ball während die Kinder zuschauen und lachen weil er so lustig ist obwohl er eigentlich müde sein sollte.",
        "word_count": 26,
        "category": "long",
    },
    {
        "text": "Der Hund läuft im Garten und spielt mit dem Ball während die Kinder zuschauen und lachen weil er so lustig ist obwohl er eigentlich müde sein sollte und gestern den ganzen Tag geschlafen hat obwohl er sonst immer актив ist und neue Tricks lernt die er dann vorführt.",
        "word_count": 45,
        "category": "very_long",
    },
]

STRESS_TEST_HIGH_DENSITY = [
    {
        "corrupted": "Das ist dem Peter sein Auto.",
        "original": "Das ist Peters Auto.",
        "category": "dialect_possessive",
        "note": "Colloquial German possessive",
    },
    {
        "corrupted": "Ich geh dann mal.",
        "original": "Ich gehe dann mal.",
        "category": "dialect_ellipsis",
        "note": "Colloquial verb dropping",
    },
    {
        "corrupted": "Der Cake ist gut.",
        "original": "Der Kuchen ist gut.",
        "category": "language_mixing",
        "note": "English word inserted",
    },
    {
        "corrupted": "Sie hat guckt der Film gestern.",
        "original": "Sie hat gestern den Film geguckt.",
        "category": "word_order",
        "note": "Heavy word order inversion",
    },
    {
        "corrupted": "Ich finde das nicht gut ich meine dass er war wrong.",
        "original": "Ich finde das nicht gut, ich meine, dass er wrong war.",
        "category": "mixed_errors",
        "note": "English insertion and word order",
    },
]

STRESS_TEST_DIALECT = [
    {
        "corrupted": "Da oben drunten ist's dunkel.",
        "original": "Da oben drinnen ist es dunkel.",
        "category": "dialect",
        "note": "Swabian/Austrian dialect form",
    },
    {
        "corrupted": "Ich hab's ihm gsagt.",
        "original": "Ich habe es ihm gesagt.",
        "category": "dialect_ellipsis",
        "note": "Bavarian contraction",
    },
    {
        "corrupted": "Des is mei Auto.",
        "original": "Das ist mein Auto.",
        "category": "dialect",
        "note": "Bavarian dialect",
    },
]


def get_all_tier_data():
    """Returns all tiered examples combined."""
    all_data = []
    for item in TIER1_EXAMPLES:
        all_data.append({"tier": 1, **item})
    for item in TIER2_EXAMPLES:
        all_data.append({"tier": 2, **item})
    for item in TIER3_EXAMPLES:
        all_data.append({"tier": 3, **item})
    return all_data


def get_flawless_data():
    """Returns all flawless sentences for FPR testing."""
    return FLAWLESS_EXAMPLES


def get_stress_test_data():
    """Returns all stress test data."""
    return {
        "length": STRESS_TEST_LENGTH,
        "high_density": STRESS_TEST_HIGH_DENSITY,
        "dialect": STRESS_TEST_DIALECT,
    }
