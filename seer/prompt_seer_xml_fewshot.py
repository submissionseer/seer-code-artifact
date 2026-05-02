from __future__ import annotations

SYSTEM_PROMPT = """You are a careful evaluator. Follow the XML schema exactly.

Formatting rules (HARD):
- Output XML only, no extra text.
- <specific_information_required>: number facts exactly f1, f2, f3, ... (no gaps).
- <present_information>: one mapping per line "fi->pj" or "fi->pj,pk"; use NONE only if no requirements are supported.
- <missing_information>: one fact ID per line, e.g., "f2"; use NONE only if nothing is missing.
- Every required fact ID must appear exactly once, in present OR missing (never both).
- Only use fact IDs listed in <specific_information_required>.
- Only cite passage IDs that exist in the provided passages.
"""

USER_PROMPT_TEMPLATE = """Given the question and retrieved passages, list the minimal disjoint requirements.
Return XML in the exact format:
<specific_information_required>
f1) ...
f2) ...
</specific_information_required>
<analysis>...</analysis>
<present_information>
f1->p2
f2->p1,p3
</present_information>
<missing_information>
f2
</missing_information>

First we will show you some examples of how to analyze a question and passages.

========================================
EXAMPLES - NEVER EVER REFERENCE THESE EXAMPLES IN YOUR RESPONSE
========================================

#### EXAMPLES ####

Example 1:
Question: What university did the creator of "The Big Bang Theory" attend?
Retrieved Passages: No search results available.

<specific_information_required>
f1) The creator of "The Big Bang Theory"
f2) The university attended by this creator
</specific_information_required>
<analysis>
To answer this question, I need to identify who created "The Big Bang Theory" and then determine which university they attended. However, the search returned no results, so I have no passages to analyze. Without any retrieved information, I cannot identify any supporting passages.
</analysis>
<present_information>
NONE
</present_information>
<missing_information>
f1
f2
</missing_information>

Example 2:
Question: What is the population of the city where the headquarters of SpaceX is located?
Retrieved Passages:
[p1] Falcon 9 | Falcon 9 is a two-stage-to-orbit medium lift launch vehicle designed and manufactured by SpaceX in the United States.
[p2] Elon Musk | Elon Reeve Musk (born June 28, 1971) is a business magnate and investor. He is the founder, CEO, and chief engineer of SpaceX.
[p3] Starlink | Starlink is a satellite internet constellation operated by SpaceX, providing satellite Internet access coverage to 40 countries.

<specific_information_required>
f1) The headquarters location of SpaceX
f2) The population of the city where SpaceX headquarters is located
</specific_information_required>
<analysis>
The question asks for the population of the city where SpaceX is headquartered. I need to first identify where SpaceX headquarters is located, then find the population of that city. Looking through the passages: [p1] discusses Falcon 9 rocket but mentions no location information, [p2] lists Elon Musk's roles including CEO of SpaceX but provides no headquarters location, and [p3] describes Starlink satellite service but again contains no location information. None of the passages contain the required facts.
</analysis>
<present_information>
NONE
</present_information>
<missing_information>
f1
f2
</missing_information>

Example 3:
Question: What year was the company that developed the PlayStation founded?
Retrieved Passages:
[p1] PlayStation | The PlayStation is a home video game console developed and marketed by Sony Interactive Entertainment. The console was released in Japan on December 3, 1994, and worldwide in 1995.
[p2] Nintendo | Nintendo Co., Ltd. is a Japanese multinational video game company headquartered in Kyoto. Nintendo was founded in 1889 by Fusajiro Yamauchi.
[p3] Xbox | Xbox is a video gaming brand created and owned by Microsoft. The brand consists of five video game consoles, as well as applications, streaming services, an online service by the name of Xbox Network, and the development arm by the name of Xbox Game Studios.

<specific_information_required>
f1) The company that developed the PlayStation
f2) The founding year of the company that developed the PlayStation
</specific_information_required>
<analysis>
This is a multi-hop question requiring two pieces of information: first identifying which company developed the PlayStation, then finding when that company was founded. Passage [p1] states that the PlayStation was developed by Sony Interactive Entertainment. However, none of the passages contain information about when Sony was founded. Passage [p2] provides founding information for Nintendo (1889) but this is not the company that developed PlayStation. Passage [p3] discusses Xbox and Microsoft but is not relevant. Only passage [p1] contains one of the required facts.
</analysis>
<present_information>
f1->p1
</present_information>
<missing_information>
f2
</missing_information>

Example 4:
Question: What nationality is the director of "Inception"?
Retrieved Passages:
[p1] Christopher Nolan | Christopher Edward Nolan (born 30 July 1970) is a British-American film director, producer, and screenwriter.
[p2] Inception | Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with Emma Thomas, his wife.
[p3] The Dark Knight | The Dark Knight is a 2008 superhero film directed, produced, and co-written by Christopher Nolan. Based on the DC Comics character Batman, the film is the second installment of Nolan's The Dark Knight Trilogy.

<specific_information_required>
f1) The director of "Inception"
f2) The nationality of the director of "Inception"
</specific_information_required>
<analysis>
To find the nationality of the director of "Inception", I need to identify who directed the film and determine their nationality. Passage [p2] states that Inception was written and directed by Christopher Nolan. Passage [p1] describes Christopher Nolan as British-American, providing his nationality. Passage [p3] mentions Nolan directed The Dark Knight but doesn't contribute required information for this question. Both passages [p1] and [p2] contain facts required to fully answer the question.
</analysis>
<present_information>
f1->p2
f2->p1
</present_information>
<missing_information>
NONE
</missing_information>

Example 5:
Question: What is the zip code of the headquarters of Spotify?
Retrieved Passages:
[p1] Spotify | Spotify is a Swedish audio streaming service headquartered in Stockholm. It offers digital rights management–protected recorded music and podcasts.
[p2] Stockholm | Stockholm is the capital and largest city of Sweden, comprising 14 islands where Lake Mälaren flows into the Baltic Sea. The city is divided into several districts and boroughs.
[p3] Daniel Ek | Daniel Ek (born 21 February 1983) is a Swedish billionaire entrepreneur and technologist. He is best known as the co-founder and CEO of Spotify.

<specific_information_required>
f1) The zip code of Spotify's headquarters
</specific_information_required>
<analysis>
This is a single-hop question asking for a specific attribute (zip code) of Spotify's headquarters. Passage [p1] states that Spotify is headquartered in Stockholm but does not provide any zip code. Passage [p2] describes Stockholm at a high level without postal details. Passage [p3] is about Daniel Ek and does not include address information. None of the passages contain the required zip code.
</analysis>
<present_information>
NONE
</present_information>
<missing_information>
f1
</missing_information>

Example 6:
Question: In which country is the city of Bruges located?
Retrieved Passages:
[p1] Bruges | Bruges is a historic city and municipality in the Flemish Region of Belgium. It is the capital and largest city of the province of West Flanders.
[p2] Amsterdam | Amsterdam is the capital and most populous city of the Netherlands, known for its canals, narrow houses, and cultural heritage.
[p3] Ghent | Ghent is a city and a municipality in the Flemish Region of Belgium. It is the capital and largest city of the East Flanders province.

<specific_information_required>
f1) The country where Bruges is located
</specific_information_required>
<analysis>
This is a single-hop question: identify the country of the city Bruges. Passage [p1] explicitly states that Bruges is in Belgium. Passages [p2] and [p3] mention other cities but do not change the answer. Therefore, the required fact is supported by [p1].
</analysis>
<present_information>
f1->p1
</present_information>
<missing_information>
NONE
</missing_information>

========================================
END OF EXAMPLES
========================================

========================================
ANALYSIS - ANALYZE ONLY THE CONTENT BELOW
========================================

NOW PLEASE ANALYZE THIS SPECIFIC QUESTION AND PASSAGES:
#### QUESTION ####
{question}

#### RETRIEVED PASSAGES ####
{passages}

CRITICAL RULES:
- Only analyze the QUESTION and PASSAGES above.
- The questions can be single or multi-hop; do not assume any specific structure or cardinality.
- Passage IDs must be from the provided passages [p1..pN].
- Every fact ID f1, f2, ... must appear exactly once (present or missing).
- Only the allowed tags listed above.
"""
