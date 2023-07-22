#per NER
from transformers import AutoTokenizer, AutoModel
import spacy

#per embedding e similarità
import torch
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

#per formattazione stringa
import urllib.parse

#per ricerca su wikipedia
import wikipedia
wikipedia.set_lang('it')
from wikipedia.exceptions import DisambiguationError


def get_entities(text):
    nlp = spacy.load("it_core_news_sm")
    doc = nlp(text)
    entities = []
    for entity in doc.ents:
        if entity.text not in entities:
            entities.append(entity.text)
    return entities

def generate_wikipedia_link(query):
    # Sostituisci gli spazi con "-"
    formatted_query = query.replace(" ", "_")
    # Codifica la stringa per gestire caratteri speciali
    encoded_query = urllib.parse.quote(formatted_query)
    # Costruisci il link completo
    wikipedia_link = f"https://it.wikipedia.org/wiki/{encoded_query}"
    return wikipedia_link

def calcola_embedding_testo_entity(text, entity):
    tokenizer = AutoTokenizer.from_pretrained("obrizum/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("obrizum/all-MiniLM-L6-v2")
    
    text_split = text.split('\n')
    embeddings = [] 
    for frase in text_split: 
        if entity in frase:
            encoded_input = tokenizer(frase, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze()
            embeddings.append(sentence_embedding)

    if len(embeddings) == 0:
        for frase in text_split:
            encoded_input = tokenizer(frase, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze()
            embeddings.append(sentence_embedding)

    embedding_testo = torch.mean(torch.stack(embeddings), dim=0).numpy()
    return embedding_testo

def calcola_migliore_similarita(entity, search_results, text):
    max_sim = 0
    best_entity = 0

    embedding1 = calcola_embedding_testo_entity(text, entity)
    embedding1 = numpy.array(embedding1).reshape(1, -1)

    link = generate_wikipedia_link(entity)

    for i,result in enumerate(search_results):
        try:
            wikilink = wikipedia.page(result).url
            if link == wikilink:
                return i;
        except wikipedia.exceptions.DisambiguationError:
            print ("pagina trovata ma disambiguazione, adesso parte il check della similarità") 


        try:
            text2 = wikipedia.summary(result)
            embedding2 = calcola_embedding_testo_entity(text2, result)
            #embedding2 = calcola_embedding_testo(text2)
        
        except wikipedia.exceptions.DisambiguationError as r:
            print("l'entità " + entity + " ha diverse ambiguità:")
            #multipla = True
            for e in r.options:
                text2 = wikipedia.summary(e)
                embedding2 = calcola_embedding_testo_entity(text2, result)
                embedding2 = numpy.array(embedding2).reshape(1, -1)

                cosine_scores = cosine_similarity(embedding1, embedding2)[0][0]

                print("\t risultato: " + entity + "    ---->ambiguità : " + e + "    ----->sim: ", cosine_scores)
                if cosine_scores > max_sim:
                    max_sim = cosine_scores
                    best_entity = i  

        cosine_scores = util.cos_sim(embedding1,embedding2)
        #if multipla == False:
            #print("entità: " + entity + "    ---->search_result: " + result + "    ----->sim: ", cosine_scores)
        #multipla = False

        if cosine_scores > max_sim:
            max_sim = cosine_scores
            best_entity = i     
    return best_entity

def get_wikipedia_pages(entities, text):
    summaries = {}
    for entity in entities:
        search_results = wikipedia.search(entity, results = 5)
        
        if len(search_results) > 0:
            bestDis = calcola_migliore_similarita(entity, search_results, text)
            result = search_results[bestDis]

            try:
                page = wikipedia.page(result)
                summaries[entity] = wikipedia.summary(result)
            #questo except serve perchè il risultato migliore può comunque generare disambiguation
            #nel caso dovesse succedere .options contiene le varie possibilità
            except wikipedia.exceptions.DisambiguationError as r:
                bestDis = calcola_migliore_similarita(entity, r.options, text)
                page = wikipedia.page(r.options[bestDis]) 
                summaries[entity] = wikipedia.summary(r.options[bestDis])

            print("\nEntity : " + entity + "\n Url:    " + page.url +  "\nSummary page: "+ summaries[entity] + "\n")

        else:
            print("Nessuna pagina wikipedia per l'entità  '" + entity + "'")    
    
    return



text = "Fatto sta che Lollobrigida è il marito di Arianna Meloni, sorella della premier - e quindi anche suo cognato, oltre che fedelissimo alleato in Fratelli d’Italia."

entities = get_entities(text)
print("Dato il seguente testo:\n", text, "\n\nLe entità trovate sono:\n")
for entity in entities:
    print(entity)

print ("\n\nPer ogni entità il programma trova il link della pagina wikipedia corrispondente all'entità. Ecco i risultati prodotti:\n\n")
get_wikipedia_pages(entities, text)