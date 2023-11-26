import os
import requests
import json

import pandas as pd
from scipy import spatial

def getWordEmbeddings(wordToEmbed: str):
    api_base_url = os.getenv('OPENAI_API_BASE')
    api_url = f"{api_base_url}openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01"
    api_key = os.getenv('OPENAI_API_KEY')
    headers = { "Content-Type":"application/json", "api-key":api_key }
    apiInput = json.dumps({ "input" : wordToEmbed })
    embResponse = requests.post(api_url, data=apiInput, headers=headers).json()['data'][0]['embedding']
    return embResponse


def get_similarity_from_text(text1, text2, varNameText1: str = "", varNameText2: str = "", printText: bool = True):
    # get embeddings for text1
    e1 = getWordEmbeddings(text1)
    # get embeddings for text2
    e2 = getWordEmbeddings(text2)
    # return cosine similarity between two embeddings 
    #s = cosine_similarity(e1, e2)
    s = 1-spatial.distance.cosine(e1, e2)

    print(f"Somiglianza tra '{text1 if printText else varNameText1}' e '{text2 if printText else varNameText2}': {s}")

def prepareDataFrame():
        cv1 = """{
        "nome": "Giovanni Bianchi",
        "sommario": "Sviluppatore .NET con più di 5 anni di esperienza nella progettazione e implementazione di soluzioni software innovative. Leader tecnico con esperienza nella gestione di progetti e nel coordinamento di team di sviluppo.",
        "esperienze_lavorative": [
            {
            "titolo": "Sviluppatore .NET Senior",
            "azienda": "ABC Solutions",
            "descrizione": "- Responsabile dello sviluppo di un sistema di gestione aziendale basato su ASP.NET Core e C#.\n- Progettazione e implementazione di architetture scalabili e robuste.\n- Coordinamento di un team di sviluppatori, fornendo linee guida tecniche e supporto.\n- Collaborazione con i clienti per comprendere le esigenze e fornire soluzioni personalizzate.\n- Ottimizzazione delle prestazioni e risoluzione di problemi complessi."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea magistrale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Linguaggi di programmazione: C#, HTML, CSS, JavaScript",
            "Framework: .NET, ASP.NET Core, Entity Framework, Xamarin",
            "Database: Microsoft SQL Server, Oracle, MongoDB",
            "Strumenti: Visual Studio, Git, Azure DevOps, Docker",
            "Conoscenza approfondita dei principi di progettazione software e dei design pattern",
            "Esperienza nell'integrazione di sistemi di terze parti e API",
            "Competenza nella gestione di progetti e nello sviluppo Agile (Scrum, Kanban)"
        ],
        "certificazioni": [
            "Certificazione Microsoft Certified: Azure Developer Associate",
            "Certificazione Microsoft Certified: .NET Developer"
        ],
        "pubblicazioni_contributi": [
            "Autore di articoli tecnici su blog e riviste specializzate",
            "Relatore in conferenze e meetup sulla programmazione .NET"
        ]
        }
        """
        cv2 = """{
        "nome": "Mario Rossi",
        "sommario": "Professionista infrastrutturale altamente qualificato con più di 5 anni di esperienza nella gestione e nell'ottimizzazione delle infrastrutture IT. Esperienza nella progettazione e implementazione di soluzioni infrastrutturali complesse.",
        "esperienze_lavorative": [
            {
            "titolo": "Responsabile infrastrutturale",
            "azienda": "ABC Solutions",
            "descrizione": "- Progettazione e implementazione di architetture di rete e infrastrutture cloud.\n- Gestione di team infrastrutturali, fornendo orientamenti tecnici e supporto.\n- Sicurezza informatica: implementazione di politiche di sicurezza, gestione degli accessi, monitoraggio delle minacce.\n- Collaborazione con i reparti di sviluppo e di supporto per la risoluzione dei problemi."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea magistrale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Sistemi operativi: Windows Server, Linux (Red Hat, Ubuntu)",
            "Networking: TCP/IP, VLAN, VPN, firewall, load balancing",
            "Cloud computing: Amazon Web Services (AWS), Microsoft Azure",
            "Sicurezza informatica: monitoraggio delle minacce, access control, penetration testing",
            "Strumenti: Active Directory, PowerShell, Ansible, Terraform",
            "Conoscenza approfondita della gestione dei progetti infrastrutturali e delle best practice di ITIL"
        ],
        "certificazioni": [
            "Certificazione Cisco Certified Network Associate (CCNA)",
            "Certificazione Microsoft Certified: Azure Administrator Associate",
            "Certificazione ITIL Foundation"
        ]
        }
        """
        cvAnalistaMag5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Analista funzionale esperto con oltre 5 anni di esperienza nella raccolta, analisi e gestione dei requisiti per progetti software complessi. Capacità dimostrate nel tradurre i requisiti aziendali in soluzioni funzionali e nel coordinare le attività di implementazione.",
        "esperienze_lavorative": [
            {
            "titolo": "Senior Business Analyst presso ABC Solutions",
            "descrizione": "- Collaborazione con gli stakeholder chiave per definire i requisiti funzionali di progetti complessi.\n- Analisi dei processi aziendali, identificazione delle lacune e proposta di soluzioni ottimali.\n- Coordinamento con il team di sviluppo per garantire il completamento dei deliverable secondo i tempi previsti.\n- Gestione del ciclo di vita dei requisiti, dalla raccolta all'implementazione e al testing."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea magistrale in Ingegneria Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Analisi dei requisiti: raccolta, documentazione e gestione dei requisiti",
            "Modellazione dei processi aziendali: diagrammi di flusso, UML",
            "Metodologie di sviluppo software: Waterfall, Agile",
            "Strumenti: Microsoft Office Suite, JIRA, Confluence",
            "Gestione dei cambiamenti e gestione degli stakeholder",
            "Conoscenza approfondita della progettazione e sviluppo software",
            "Competenze di problem-solving e pensiero critico"
        ],
        "certificazioni": [
            "Certificazione IIBA Certified Business Analysis Professional (CBAP)",
            "Certificazione Agile Certified Practitioner (PMI-ACP)"
        ]
        }
        """

        cvSvilMin5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Sviluppatore .NET altamente motivato con solide competenze di programmazione e problem-solving. Orientato all'apprendimento continuo e alla collaborazione per raggiungere risultati di successo.",
        "esperienze_lavorative": [
            {
            "titolo": "Stage presso ABC Solutions",
            "descrizione": "- Sviluppo di applicazioni basate su .NET Framework utilizzando C#.\n- Collaborazione con il team di sviluppo per la realizzazione di funzionalità software.\n- Test e risoluzione dei problemi."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea triennale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Linguaggi di programmazione: C#, HTML, CSS, JavaScript",
            "Framework: .NET Framework, ASP.NET, Entity Framework",
            "Database: Microsoft SQL Server, MySQL",
            "Strumenti: Visual Studio, Git",
            "Conoscenza dei concetti di base di progettazione e sviluppo software"
        ]
        }
        """
        cvInfraMin5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Professionista infrastrutturale con passione per la tecnologia e solide competenze nella gestione delle infrastrutture IT. Orientato alla risoluzione dei problemi e al miglioramento continuo.",
        "esperienze_lavorative": [
            {
            "titolo": "Tecnico infrastrutturale presso XYZ Corporation",
            "descrizione": "- Configurazione e manutenzione di server e reti aziendali.\n- Supporto tecnico agli utenti e risoluzione dei problemi infrastrutturali.\n- Monitoraggio delle prestazioni e ottimizzazione delle risorse di rete."
            }
        ],
        "istruzione": [
            {
            "titolo": "Diploma tecnico in Informatica",
            "istituto": "Scuola ABC"
            }
        ],
        "competenze_tecniche": [
            "Sistemi operativi: Windows Server, Linux",
            "Networking: TCP/IP, DNS, DHCP",
            "Sicurezza informatica: firewall, antivirus, backup",
            "Strumenti: Active Directory, VMware, Wireshark",
            "Competenze di base nella gestione dei progetti"
        ]
        }
        """
        cvAnalistaMin5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Analista funzionale motivato con solide competenze analitiche e di problem-solving. Capacità di comprendere le esigenze aziendali e tradurle in specifiche funzionali.",
        "esperienze_lavorative": [
            {
            "titolo": "Junior Analyst presso ABC Solutions",
            "descrizione": "- Raccolta e analisi dei requisiti funzionali in progetti software.\n- Supporto nella progettazione di soluzioni e nella stesura di documentazione.\n- Collaborazione con il team di sviluppo per garantire l'allineamento alle specifiche."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea triennale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Analisi dei requisiti: raccolta, documentazione e gestione dei requisiti",
            "Modellazione dei processi aziendali: diagrammi di flusso, UML",
            "Metodologie di sviluppo software: Waterfall, Agile",
            "Strumenti: Microsoft Office Suite, JIRA",
            "Conoscenza dei principi di base dell'analisi funzionale"
        ]
        }
        """


        cvSvil5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Sviluppatore .NET con 5 anni di esperienza nella progettazione e implementazione di soluzioni software. Esperto nella scrittura di codice pulito e mantenibile, con una solida comprensione dei principi di sviluppo software.",
        "esperienze_lavorative": [
            {
            "titolo": "Sviluppatore .NET presso ABC Solutions",
            "descrizione": "- Progettazione e sviluppo di applicazioni aziendali basate su .NET utilizzando C#.\n- Collaborazione con il team di sviluppo per la realizzazione di funzionalità complesse.\n- Migrazione di applicazioni legacy a nuove architetture tecnologiche.\n- Implementazione di test automatici per garantire la qualità del software."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea magistrale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Linguaggi di programmazione: C#, HTML, CSS, JavaScript",
            "Framework: .NET, ASP.NET MVC, Entity Framework",
            "Database: Microsoft SQL Server, MySQL",
            "Strumenti: Visual Studio, Git, Azure DevOps",
            "Conoscenza approfondita dei principi di progettazione software e dei design pattern",
            "Esperienza nello sviluppo di applicazioni Web e servizi RESTful",
            "Competenza nella gestione di progetti e nello sviluppo Agile (Scrum)"
        ],
        "certificazioni": [
            "Certificazione Microsoft Certified: Azure Developer Associate",
            "Certificazione Microsoft Certified: .NET Developer"
        ]
        }
        """
        cvInfra5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Professionista infrastrutturale con 5 anni di esperienza nella gestione e nell'ottimizzazione delle infrastrutture IT. Competenze avanzate nella progettazione e implementazione di soluzioni infrastrutturali complesse.",
        "esperienze_lavorative": [
            {
            "titolo": "Responsabile infrastrutturale presso ABC Solutions",
            "descrizione": "- Progettazione e implementazione di architetture di rete e infrastrutture cloud.\n- Gestione di team infrastrutturali e coordinamento delle attività.\n- Monitoraggio delle prestazioni e risoluzione dei problemi infrastrutturali.\n- Implementazione di soluzioni per la sicurezza e il disaster recovery."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea triennale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Sistemi operativi: Windows Server, Linux (Red Hat, Ubuntu)",
            "Networking: TCP/IP, VLAN, VPN, firewall, load balancing",
            "Cloud computing: Amazon Web Services (AWS), Microsoft Azure",
            "Sicurezza informatica: monitoraggio delle minacce, access control, penetration testing",
            "Strumenti: Active Directory, PowerShell, Ansible, Terraform",
            "Conoscenza approfondita della gestione dei progetti infrastrutturali e delle best practice di ITIL"
        ],
        "certificazioni": [
            "Certificazione Cisco Certified Network Professional (CCNP)",
            "Certificazione Microsoft Certified: Azure Solutions Architect Expert",
            "Certificazione ITIL Intermediate"
        ]
        }
        """
        cvAnalista5Anni = """{
        "nome": "Nome Cognome",
        "indirizzo": "Indirizzo",
        "telefono": "Numero di telefono",
        "email": "Indirizzo email",
        "sommario": "Analista funzionale con 5 anni di esperienza nella raccolta e analisi dei requisiti per progetti software complessi. Abilità nell'interfacciarsi con gli stakeholder e nella traduzione dei requisiti in specifiche funzionali.",
        "esperienze_lavorative": [
            {
            "titolo": "Analista funzionale presso ABC Solutions",
            "descrizione": "- Collaborazione con team multidisciplinari per definire i requisiti funzionali di progetti software.\n- Analisi dei processi aziendali e identificazione delle aree di miglioramento.\n- Stesura di documenti di analisi funzionale e specifiche dei requisiti.\n- Supporto al team di sviluppo durante la fase di implementazione."
            }
        ],
        "istruzione": [
            {
            "titolo": "Laurea magistrale in Informatica",
            "universita": "Università XYZ"
            }
        ],
        "competenze_tecniche": [
            "Analisi dei requisiti: raccolta, documentazione e gestione dei requisiti",
            "Modellazione dei processi aziendali: diagrammi di flusso, UML",
            "Metodologie di sviluppo software: Waterfall, Agile",
            "Strumenti: Microsoft Office Suite, JIRA, Confluence",
            "Esperienza nella gestione dei cambiamenti e nella gestione degli stakeholder",
            "Conoscenza dei principi di base della progettazione e dello sviluppo software"
        ],
        "certificazioni": [
            "Certificazione IIBA Certified Business Analysis Professional (CBAP)",
            "Certificazione Agile Certified Practitioner (PMI-ACP)"
        ]
        }
        """

        # Create DataFrame with all the cv dictionary
        cvRepo = { 'CV_Description' : ["CV Sviluppatore > 5anni", "CV Infrastrutturale > 5anni", "CV Analista > 5anni", "CV Sviluppatore < 5", "CV Infrastrutturale < 5", "CV Analista < 5",
                                    "CV Sviluppatore 5anni", "CV Infrastrutturale 5anni", "CV Analista 5anni"],
                'CV' : [cv1, cv2, cvAnalistaMag5Anni, cvSvil5Anni, cvInfra5Anni, cvAnalista5Anni, cvSvilMin5Anni, cvInfraMin5Anni, cvAnalistaMin5Anni]}
        dfCvRepo = pd.DataFrame(cvRepo)

        # add embedding column with embedding
        dfCvRepo['embedding'] = dfCvRepo['CV'].apply(lambda x: getWordEmbeddings(x))
        return dfCvRepo

# populate CV repository
CVRepo = prepareDataFrame()

# Let's make a search on libraries df based on a particular need
HRNeeds = "Sviluppatore .net con almeno cinque anni di esperienza"

requestEmbedded = getWordEmbeddings(HRNeeds)

# get cosine similarity between request and all cv descriptions in DataFrame
similarities = CVRepo['embedding'].apply(lambda x: 1 - spatial.distance.cosine(x, requestEmbedded))
# dfCvRepo.head()

# combine CV description from request and similiaries and sort ascending by similarity
recommendations = pd.concat([CVRepo['CV_Description'], similarities], axis=1).sort_values(by='embedding', ascending=False)
print(recommendations.head(10))