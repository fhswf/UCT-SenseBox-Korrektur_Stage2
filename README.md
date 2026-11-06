# KI-Korrektur Stage 2 – Urban Climate Twin Soest

Die Stage 2 KI-Korrektur liefert korrigierte Temperaturmessungen von SenseBoxen in Form eines nachträglichen Datenbank-Updates. Ein trainiertes Machine-Learning-Modell nutzt Messungen benachbarter Stationen, DWD-Daten und weitere Merkmale, um korrigierte Werte zu bestimmen und in der Datenbank zu speichern.

## Urban Climate Twin Soest

 [Urban Climate Twin (UCT) Soest](https://www.fh-swf.de/de/forschung___transfer_4/forschungsprojekte_1/forschungsprojekt_uct.php) ist ein digitaler Zwilling des Soester Stadtklimas. Er führt Messdaten, Geodaten und weitere Klimadaten zusammen und macht das Stadklima räumlich nachvollziehbar. Bürger und die Stadtverwaltung können so die Auswirkungen von Klimaveränderungen auf das Stadtgebiet besser verstehen und Maßnahmen zur Anpassung an den Klimawandel entwickeln.

## Funktionsweise

Das Skript `main.py` lädt Messungen aus der konfigurierten PostgreSQL-Datenbank und bereitet sie für die Stage-2-Korrektur auf. Je nach Konfiguration werden Nachbar-SenseBoxen und DWD-Messungen zugeordnet, JSON- und Rasterdaten extrahiert, fehlende Werte behandelt und Zeitmerkmale erzeugt.

Anschließend lädt das Skript das trainierte Modell und den zugehörigen Scaler aus `model_files/`, berechnet die korrigierten Temperaturwerte und schreibt sie zurück in die Datenbank. Außerdem wird der Zeitstempel der letzten Korrektur in der Datenbank aktualisiert.

## Voraussetzungen

 - Python 3.12+
 - Die in `requirements.txt` aufgeführten Python-Pakete

## Konfiguration

1. `.env.example` in `.env` umbenennen und Zugangsdaten der Datenbank anpassen.

2. In `config.yaml` Datenbankname und Schema, Tabellen, Zeitraum, Vorverarbeitung sowie Modell- und Scaler-Pfade prüfen. 

Die Verarbeitung wird in `logs/stage2.log` protokolliert.

## Projektstruktur

```text
.
├── main.py                 # Lädt Daten, führt Stage-2-Inferenz aus und schreibt Ergebnisse
├── config.yaml             # Datenbank-, Vorverarbeitungs- und Modelleinstellungen
├── .env.example            # Vorlage für Datenbank-Zugangsdaten
├── requirements.txt        # Python-Abhängigkeiten
├── utils.py                # Hilfsfunktionen
├── preprocessing/          # Datenextraktion, Vorverarbeitung
├── model/
│   └── build_model.py      # Definition des KI-Modells
├── model_files/            # Modellgewichte und Scaler
└── logs/                   # Protokolle
```

## Lizenz

Copyright 2026 Urban Climate Twin, Fachhochschule Südwestfalen

Lizenziert unter der [Apache License 2.0](LICENSE.md).