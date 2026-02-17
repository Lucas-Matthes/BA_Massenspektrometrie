# BA Massenspektrometer Pipeline

Dieses Projekt enthält die Pipeline zur Analyse von Massenspektrometerdaten:
- Feature-Extraktion aus Excel-Dateien
- Random Forest Klassifikation
- DBSCAN Clustering (inkl. dynamische Distanzmetriken wie Jensen-Shannon)
- Compound Classifier

## Voraussetzungen

- Python 3.10+ (oder 3.14, wie im Projekt)

## Installs

pip install -r requirements.txt

## Ausführen

### Standard (empfohlen):

python -m src.MasterPipeline
